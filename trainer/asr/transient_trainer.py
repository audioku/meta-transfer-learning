import time
import numpy as np
import torch
import logging
import sys
import threading
import time

from copy import deepcopy
from tqdm import tqdm
# from utils import constant
from collections import deque
from utils.functions import save_meta_model, post_process
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer
from torch.autograd import Variable

class TransientTrainer():
    """
    Trainer class
    """
    def __init__(self):
        logging.info("Transient Trainer is initialized")

    def forward_one_batch(self, model, vocab, src, trg, src_percentages, src_lengths, trg_lengths, smoothing, loss_type, verbose=False):
        pred, gold, hyp = model(src, src_lengths, trg, verbose=False)
        strs_golds, strs_hyps = [], []

        for j in range(len(gold)):
            ut_gold = gold[j]
            strs_golds.append("".join([vocab.id2label[int(x)] for x in ut_gold]))
        
        for j in range(len(hyp)):
            ut_hyp = hyp[j]
            strs_hyps.append("".join([vocab.id2label[int(x)] for x in ut_hyp]))

        # handling the last batch
        seq_length = pred.size(1)
        sizes = src_percentages.mul_(int(seq_length)).int()

        loss, _ = calculate_metrics(pred, gold, vocab.PAD_ID, input_lengths=sizes, target_lengths=trg_lengths, smoothing=smoothing, loss_type=loss_type)

        if loss is None:
            print("loss is None")

        if loss.item() == float('Inf'):
            logging.info("Found infinity loss, masking")
            print("Found infinity loss, masking")
            loss = torch.where(loss != loss, torch.zeros_like(loss), loss) # NaN masking

        # if verbose:
        #     print(">PRED:", strs_hyps)
        #     print(">GOLD:", strs_golds)

        total_cer, total_wer, total_char, total_word = 0, 0, 0, 0
        for j in range(len(strs_hyps)):
            strs_hyps[j] = post_process(strs_hyps[j], vocab.special_token_list)
            strs_golds[j] = post_process(strs_golds[j], vocab.special_token_list)
            cer = calculate_cer(strs_hyps[j].replace(' ', ''), strs_golds[j].replace(' ', ''))
            wer = calculate_wer(strs_hyps[j], strs_golds[j])
            total_cer += cer
            total_wer += wer
            total_char += len(strs_golds[j].replace(' ', ''))
            total_word += len(strs_golds[j].split(" "))
        
        if verbose:
            print('Total CER', total_cer)
            print('Total char', total_char)

            print("PRED:", strs_hyps)
            print("GOLD:", strs_golds, flush=True)

        return loss, total_cer, total_char

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train(self, model, vocab, train_data_list, valid_loader_list, loss_type, start_it, num_it, args, inner_opt=None, outer_opt=None, evaluate_every=1000, window_size=100, last_summary_every=1000, last_metrics=None, early_stop=10, cpu_state_dict=False, is_copy_grad=False):
        """
        Training
        args:
            model: Model object
            train_data_list: DataLoader object of the training set
            valid_loader_list: a list of Validation DataLoader objects
            start_it: start it (> 0 if you resume the process)
            num_it: last epoch
            last_metrics: (if resume)
        """
        history = []
        best_valid_val = 1000000000
        smoothing = args.label_smoothing
        early_stop_criteria, early_stop_val = early_stop.split(",")[0], int(early_stop.split(",")[1])
        count_stop = 0

        logging.info("name " +  args.name)

        total_time = 0

        logging.info("TRAIN")
        print("TRAIN")
        model.train()

        # define the optimizer
        if inner_opt is None:
            inner_opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        
        if outer_opt is None:
            outer_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)

        last_sum_loss = deque(maxlen=window_size)
        last_sum_cer = deque(maxlen=window_size)
        last_sum_char = deque(maxlen=window_size)
        
        # Define local variables
        k_train, k_valid = args.k_train, args.k_valid
        train_data_buffer = [[] for manifest_id in range(len(train_data_list))]
        
        # Define batch loader function
        def fetch_train_batch(train_data_list, k_train, k_valid, train_buffer):
            for manifest_id in range(len(train_data_list)):
                batch_data = train_data_list[manifest_id].sample(k_train, k_valid, manifest_id)
                train_buffer[manifest_id].insert(0, batch_data)
            return train_buffer

         # Parallelly fetch next batch data from all manifest
        prefetch = threading.Thread(target=fetch_train_batch, 
                        args=([train_data_list, k_train, k_valid, train_data_buffer]))
        prefetch.start()
        
        it = start_it
        while it < num_it:
             # Wait until the next batch data is ready
            prefetch.join()
            
            # Parallelly fetch next batch data from all manifest
            prefetch = threading.Thread(target=fetch_train_batch, 
                            args=([train_data_list, k_train, k_valid, train_data_buffer]))
            prefetch.start()
            
            # Buffer for accumulating loss
            batch_loss = 0
            total_loss, total_cer = 0, 0
            total_char = 0
                
            # Local variables
            weights_original = None
            train_tmp_buffer = None
                        
            try:
                # Start execution time
                start_time = time.time()

                # Prepare model state dict (Based on experiment it doesn't yield any difference)
                if cpu_state_dict:
                    model.cpu()
                    weights_original = deepcopy(model.state_dict())
                    model.cuda()
                else:
                    weights_original = deepcopy(model.state_dict())

                # Reinit outer opt
                outer_opt.zero_grad()
                if is_copy_grad:
                    model.zero_copy_grad() # initialize copy_grad with 0

                # Retrieve target validation
                _, val_data = train_data_buffer[-1][-1]
                val_inputs, val_input_sizes, val_percentages, val_targets, val_target_sizes = val_data
                    
                # Pop buffer for all manifest first
                # so we can maintain the same number in the buffer list if exception occur
                train_tmp_buffer = []
                for manifest_id in range(len(train_data_buffer)):  
                    train_tmp_buffer.insert(0, train_data_buffer[manifest_id].pop())
                        
                # Loop over all tasks
                for manifest_id in range(len(train_tmp_buffer)):                
                    # Retrieve manifest data
                    tr_data, _ = train_tmp_buffer.pop()
                    tr_inputs, tr_input_sizes, tr_percentages, tr_targets, tr_target_sizes = tr_data
                    if args.cuda:
                        tr_inputs = tr_inputs.cuda()
                        tr_targets = tr_targets.cuda()

                    # Meta Train
                    model.train()
                    tr_loss, tr_cer, tr_num_char = self.forward_one_batch(model, vocab, tr_inputs, tr_targets, tr_percentages, tr_input_sizes, tr_target_sizes, smoothing, loss_type, verbose=False)

                    # Update train evaluation metric                    
                    total_cer += tr_cer
                    total_char += tr_num_char

                    # Delete unused references
                    del tr_inputs, tr_input_sizes, tr_percentages, tr_targets, tr_target_sizes, tr_data

                    # Inner Backward
                    inner_opt.zero_grad()
                    tr_loss.backward()
                    
                    # Delete unused references
                    del tr_loss
                    
                    # Inner Update
                    if args.clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                    inner_opt.step()

                    # Move validation to cuda
                    if args.cuda:
                        val_cuda_inputs = val_inputs.cuda()
                        val_cuda_targets = val_targets.cuda()

                    # Meta Validation 
                    val_loss, val_cer, val_num_char = self.forward_one_batch(model, vocab, val_cuda_inputs, val_cuda_targets, val_percentages, val_input_sizes, val_target_sizes, smoothing, loss_type)

                    # batch_loss += val_loss
                    total_loss += val_loss.item()
                    
                    # # Delete unused references
                    # del val_inputs, val_input_sizes, val_percentages, val_targets, val_target_sizes, val_data
                    # del val_cuda_inputs, val_cuda_targets
                    
                    # outer loop optimization
                    if is_copy_grad:
                        val_loss = val_loss / len(train_data_list)
                        val_loss.backward()

                        model.add_copy_grad() # add model grad to copy grad
                    else:
                        batch_loss += val_loss / len(train_data_list)

                    # Delete unused references
                    del val_loss
                    
                    # Reset Weight
                    model.load_state_dict(weights_original)

                # Delete unused references
                del val_inputs, val_input_sizes, val_percentages, val_targets, val_target_sizes, val_data
                del val_cuda_inputs, val_cuda_targets
                
                # Delete copy weight
                weights_original = None
                
                # Outer loop optimization
                if is_copy_grad:
                    model.from_copy_grad() # copy grad from copy_grad to model
                else:
                    batch_loss.backward()
                    del batch_loss
                
                if args.clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                outer_opt.step()

                
                # Record performance
                last_sum_cer.append(total_cer)
                last_sum_char.append(total_char)
                last_sum_loss.append(total_loss/len(train_data_list))

                # Record execution time
                end_time = time.time()
                diff_time = end_time - start_time
                total_time += diff_time

                print("(Iteration {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f} TOTAL TIME:{:.7f}".format(
                    (it+1), total_loss/len(train_data_list), total_cer*100/total_char, self.get_lr(outer_opt), total_time))         
                logging.info("(Iteration {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f} TOTAL TIME:{:.7f}".format(
                    (it+1), total_loss/len(train_data_list), total_cer*100/total_char, self.get_lr(outer_opt), total_time))

                if (it + 1) % last_summary_every == 0:
                    print("(Summary Iteration {} | MA {}) TRAIN LOSS:{:.4f} CER:{:.2f}%".format(
                        (it+1), window_size, sum(last_sum_loss)/len(last_sum_loss), sum(last_sum_cer)*100/sum(last_sum_char)), flush=True)
                    logging.info("(Summary Iteration {} | MA {}) TRAIN LOSS:{:.4f} CER:{:.2f}%".format(
                        (it+1), window_size, sum(last_sum_loss)/len(last_sum_loss), sum(last_sum_cer)*100/sum(last_sum_char)))

                # VALID
                if (it + 1) % evaluate_every == 0:
                    print("")
                    logging.info("VALID")
                    model.eval()

                    final_valid_losses = []
                    final_valid_cers = []
                    with torch.no_grad():
                        for ind in range(len(valid_loader_list)):
                            valid_loader = valid_loader_list[ind]

                            total_valid_loss, total_valid_cer, total_valid_wer, total_valid_char, total_valid_word = 0, 0, 0, 0, 0
                            valid_pbar = tqdm(iter(valid_loader), leave=True, total=len(valid_loader))
                            for i, (data) in enumerate(valid_pbar):
                                # torch.cuda.empty_cache()
                                src, trg, src_percentages, src_lengths, trg_lengths = data
                                # try:
                                if args.cuda:
                                    src = src.cuda()
                                    trg = trg.cuda()
                                loss, cer, num_char = self.forward_one_batch(model, vocab, src, trg, src_percentages, src_lengths, trg_lengths, smoothing, loss_type)
                                total_valid_cer += cer
                                total_valid_char += num_char

                                total_valid_loss += loss.item()
                                valid_pbar.set_description("(Iteration {}) VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format((it+1), ind,
                                    total_valid_loss/(i+1), total_valid_cer*100/total_valid_char))
                                
                                del src, trg, src_percentages, src_lengths, trg_lengths

                            final_valid_loss = total_valid_loss/(len(valid_loader))
                            final_valid_cer = total_valid_cer*100/total_valid_char

                            final_valid_losses.append(final_valid_loss)
                            final_valid_cers.append(final_valid_cer)
                            print("(Iteration {}) VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format((it+1), ind, final_valid_loss, final_valid_cer))
                            logging.info("(Iteration {}) VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format((it+1), ind, final_valid_loss, final_valid_cer))

                            del total_valid_loss, total_valid_cer, total_valid_wer, total_valid_char, total_valid_word 

                    metrics = {}
                    avg_valid_loss = sum(final_valid_losses) / len(final_valid_losses)
                    avg_valid_cer = sum(final_valid_cers) / len(final_valid_cers)
                    metrics["avg_valid_loss"] = sum(final_valid_losses) / len(final_valid_losses)
                    metrics["avg_valid_cer"] = sum(final_valid_cers) / len(final_valid_cers)
                    metrics["valid_loss"] = final_valid_losses
                    metrics["valid_cer"] = final_valid_cers
                    metrics["history"] = history
                    history.append(metrics)

                    print("(Iteration {}) AVG VALID LOSS:{:.4f} AVG CER:{:.2f}%".format((it+1), sum(final_valid_losses) / len(final_valid_losses), sum(final_valid_cers) / len(final_valid_cers)))
                    logging.info("(Iteration {}) AVG VALID LOSS:{:.4f} AVG CER:{:.2f}%".format((it+1), sum(final_valid_losses) / len(final_valid_losses), sum(final_valid_cers) / len(final_valid_cers)))

                    if (it+1) % args.save_every == 0:
                        save_meta_model(model, vocab, (it+1), inner_opt, outer_opt, metrics, args, best_model=False)

                    # save the best model
                    early_stop_criteria, early_stop_val
                    if early_stop_criteria == "cer":
                        print("CRITERIA: CER")
                        if best_valid_val > avg_valid_cer:
                            count_stop = 0
                            best_valid_val = avg_valid_cer
                            save_meta_model(model, vocab, (it+1), inner_opt, outer_opt, metrics, args, best_model=True)
                        else:
                            print("count_stop:", count_stop)
                            count_stop += 1
                    else:
                        print("CRITERIA: LOSS")
                        if best_valid_val > avg_valid_loss:
                            count_stop = 0
                            best_valid_val = avg_valid_loss
                            save_meta_model(model, vocab, (it+1), inner_opt, outer_opt, metrics, args, best_model=True)
                        else:
                            count_stop += 1
                            print("count_stop:", count_stop)

                    if count_stop >= early_stop_val:
                        logging.info("EARLY STOP")
                        print("EARLY STOP\n")
                        break
                
                # Increment iteration
                it += 1
            except KeyboardInterrupt:
                raise
            except Exception as e:
                print('Error: {}, fetching new data...'.format(e), flush=True)
                logging.info('Error: {}, fetching new data...'.format(e))

                tr_inputs, tr_input_sizes, tr_percentages, tr_targets, tr_target_sizes = None, None, None, None, None
                val_inputs, val_input_sizes, val_percentages, val_targets, val_target_sizes = None, None, None, None, None       
                val_cuda_inputs, val_cuda_targets = None, None
                tr_loss, val_loss = None, None
                weights_original = None
                batch_loss = 0
        
                torch.cuda.empty_cache()