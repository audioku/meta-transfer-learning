import time
import numpy as np
import torch
import logging
import sys

from tqdm import tqdm
# from utils import constant
from utils.functions import save_model, post_process
from utils.optimizer import NoamOpt
from utils.metrics import calculate_metrics, calculate_cer, calculate_wer
from torch.autograd import Variable

class Trainer():
    """
    Trainer class
    """
    def __init__(self):
        logging.info("Trainer is initialized")

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def train_one_batch(self, model, vocab, src, trg, src_percentages, src_lengths, trg_lengths, smoothing, loss_type):
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

        loss, num_correct = calculate_metrics(pred, gold, vocab.PAD_ID, input_lengths=sizes, target_lengths=trg_lengths, smoothing=smoothing, loss_type=loss_type)

        if loss is None:
            print("loss is None")

        if loss.item() == float('Inf'):
            logging.info("Found infinity loss, masking")
            print("Found infinity loss, masking")
            loss = torch.where(loss != loss, torch.zeros_like(loss), loss) # NaN masking

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

        return loss, total_cer, total_char

    def train(self, model, vocab, train_loader, valid_loader_list, loss_type, start_epoch, num_epochs, args, evaluate_every=1000, last_metrics=None, early_stop=10, opt_name="adam"):
        """
        Training
        args:
            model: Model object
            train_loader: DataLoader object of the training set
            valid_loader_list: a list of Validation DataLoader objects
            start_epoch: start epoch (> 0 if you resume the process)
            num_epochs: last epoch
        """
        history = []
        best_valid_val = 1000000000
        smoothing = args.label_smoothing
        early_stop_criteria, early_stop_val = early_stop.split(",")[0], int(early_stop.split(",")[1])
        count_stop = 0

        logging.info("name " +  args.name)

        if opt_name == "adam":
            opt = torch.optim.Adam(model.parameters(), lr=args.lr)
        elif opt_name == "sgd":
            opt = torch.optim.SGD(model.parameters(), lr=args.lr)
        else:
            opt = None

        for epoch in range(start_epoch, num_epochs):
            total_loss, total_cer, total_wer, total_char, total_word = 0, 0, 0, 0, 0
            total_time = 0

            start_iter = 0
            final_train_losses = []
            final_train_cers = []

            logging.info("TRAIN")
            print("TRAIN")
            model.train()
            pbar = tqdm(iter(train_loader), leave=True, total=len(train_loader))
            max_len = 0
            for i, (data) in enumerate(pbar, start=start_iter):
                torch.cuda.empty_cache()
                src, trg, src_percentages, src_lengths, trg_lengths = data
                max_len = max(max_len, src.size(-1))

                opt.zero_grad()

                try:
                    if args.cuda:
                        src = src.cuda()
                        trg = trg.cuda()

                    start_time = time.time()
                    loss, cer, num_char = self.train_one_batch(model, vocab, src, trg, src_percentages, src_lengths, trg_lengths, smoothing, loss_type)
                    total_cer += cer
                    total_char += num_char
                    loss.backward()

                    if args.clip:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                    
                    opt.step()
                    total_loss += loss.item()

                    end_time = time.time()
                    diff_time = end_time - start_time
                    total_time += diff_time

                    pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f} TOTAL TIME:{:.7f}".format(
                        (epoch+1), total_loss/(i+1), total_cer*100/total_char, self.get_lr(opt), total_time))
                except Exception as e:
                    print(e)
                    # del loss
                    try:
                        torch.cuda.empty_cache()
                        src = src.cpu()
                        trg = trg.cpu()
                        src_splits, src_lengths_splits, trg_lengths_splits, trg_splits, src_percentages_splits = iter(src.split(2, dim=0)), iter(src_lengths.split(2, dim=0)), iter(trg_lengths.split(2, dim=0)), iter(trg.split(2, dim=0)), iter(src_percentages.split(2, dim=0))
                        j = 0

                        start_time = time.time()
                        for src, trg, src_lengths, trg_lengths, src_percentages in zip(src_splits, trg_splits, src_lengths_splits, trg_lengths_splits, src_percentages_splits):
                            opt.zero_grad()
                            torch.cuda.empty_cache()
                            if args.cuda:
                                src = src.cuda()
                                trg = trg.cuda()

                            start_time = time.time()
                            loss, cer, num_char = self.train_one_batch(model, vocab, src, trg, src_percentages, src_lengths, trg_lengths, smoothing, loss_type)
                            total_cer += cer
                            total_char += num_char
                            loss.backward()

                            if args.clip:
                                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                            
                            opt.step()
                            total_loss += loss.item()
                            j += 1

                        end_time = time.time()
                        diff_time = end_time - start_time
                        total_time += diff_time
                        logging.info("probably OOM, autosplit batch. succeeded")
                        print("probably OOM, autosplit batch. succeeded")
                    except:
                        logging.info("probably OOM, autosplit batch. skip batch")
                        print("probably OOM, autosplit batch. skip batch")
                        continue

            pbar.set_description("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f} TOTAL TIME:{:.7f}".format((epoch+1), total_loss/(i+1), total_cer*100/total_char, self.get_lr(opt), total_time))

            final_train_loss = total_loss/(len(train_loader))
            final_train_cer = total_cer*100/total_char

            final_train_losses.append(final_train_loss)
            final_train_cers.append(final_train_cer)

            logging.info("(Epoch {}) TRAIN LOSS:{:.4f} CER:{:.2f}% LR:{:.7f}".format(
                (epoch+1), final_train_loss, final_train_cer, self.get_lr(opt)))

            # evaluate
            if (epoch + 1) % evaluate_every == 0:
                print("")
                logging.info("VALID")
                model.eval()

                final_valid_losses = []
                final_valid_cers = []
                for ind in range(len(valid_loader_list)):
                    valid_loader = valid_loader_list[ind]

                    total_valid_loss, total_valid_cer, total_valid_wer, total_valid_char, total_valid_word = 0, 0, 0, 0, 0
                    valid_pbar = tqdm(iter(valid_loader), leave=True, total=len(valid_loader))
                    for i, (data) in enumerate(valid_pbar):
                        torch.cuda.empty_cache()

                        src, trg, src_percentages, src_lengths, trg_lengths = data
                        try:
                            if args.cuda:
                                src = src.cuda()
                                trg = trg.cuda()
                            loss, cer, num_char = self.train_one_batch(model, vocab, src, trg, src_percentages, src_lengths, trg_lengths, smoothing, loss_type)
                            total_valid_cer += cer
                            total_valid_char += num_char

                            total_valid_loss += loss.item()
                            valid_pbar.set_description("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind,
                                total_valid_loss/(i+1), total_valid_cer*100/total_valid_char))
                            # valid_pbar.set_description("(Epoch {}) VALID LOSS:{:.4f} CER:{:.2f}% WER:{:.2f}%".format(
                                # (epoch+1), total_valid_loss/(i+1), total_valid_cer*100/total_valid_char, total_valid_wer*100/total_valid_word))
                        except:
                            try:
                                torch.cuda.empty_cache()
                                src = src.cpu()
                                trg = trg.cpu()
                                src_splits, src_lengths_splits, trg_lengths_splits, trg_splits, trg_transcript_splits, src_percentages_splits = iter(src.split(2, dim=0)), iter(src_lengths.split(2, dim=0)), iter(trg_lengths.split(2, dim=0)), iter(trg.split(2, dim=0)), iter(trg_transcript.split(2, dim=0)), iter(src_percentages.split(2, dim=0))
                                j = 0
                                for src, trg, src_lengths, trg_lengths, src_percentages in zip(src_splits, trg_splits, src_lengths_splits, trg_lengths_splits, src_percentages_splits):
                                    opt.zero_grad()
                                    torch.cuda.empty_cache()
                                    if args.cuda:
                                        src = src.cuda()
                                        trg = trg.cuda()

                                    loss, cer, num_char = self.train_one_batch(model, vocab, src, trg, src_percentages, src_lengths, trg_lengths, smoothing, loss_type)
                                    total_valid_cer += cer
                                    total_valid_char += num_char

                                    if args.clip:
                                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
                                    
                                    total_valid_loss += loss.item()
                                    j += 1
                                valid_pbar.set_description("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind, total_valid_loss/(i+1), total_valid_cer*100/total_valid_char))

                                logging.info("probably OOM, autosplit batch. succeeded")
                                print("probably OOM, autosplit batch. succeeded")
                            except:
                                logging.info("probably OOM, autosplit batch. skip batch")
                                print("probably OOM, autosplit batch. skip batch")
                                continue

                    final_valid_loss = total_valid_loss/(len(valid_loader))
                    final_valid_cer = total_valid_cer*100/total_valid_char

                    final_valid_losses.append(final_valid_loss)
                    final_valid_cers.append(final_valid_cer)
                    print("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind, final_valid_loss, final_valid_cer))
                    logging.info("VALID SET {} LOSS:{:.4f} CER:{:.2f}%".format(ind, final_valid_loss, final_valid_cer))

                metrics = {}
                avg_valid_loss = sum(final_valid_losses) / len(final_valid_losses)
                avg_valid_cer = sum(final_valid_cers) / len(final_valid_cers)
                metrics["avg_train_loss"] = sum(final_train_losses) / len(final_train_losses)
                metrics["avg_valid_loss"] = sum(final_valid_losses) / len(final_valid_losses)
                metrics["avg_train_cer"] = sum(final_train_cers) / len(final_train_cers)
                metrics["avg_valid_cer"] = sum(final_valid_cers) / len(final_valid_cers)
                metrics["train_loss"] = final_train_losses
                metrics["valid_loss"] = final_valid_losses
                metrics["train_cer"] = final_train_cers
                metrics["valid_cer"] = final_valid_cers
                metrics["history"] = history
                history.append(metrics)

                print("AVG VALID LOSS:{:.4f} AVG CER:{:.2f}%".format(sum(final_valid_losses) / len(final_valid_losses), sum(final_valid_cers) / len(final_valid_cers)))
                logging.info("AVG VALID LOSS:{:.4f} AVG CER:{:.2f}%".format(sum(final_valid_losses) / len(final_valid_losses), sum(final_valid_cers) / len(final_valid_cers)))

                if epoch % args.save_every == 0:
                    save_model(model, vocab, (epoch+1), opt, metrics, args, best_model=False)

                # save the best model
                early_stop_criteria, early_stop_val
                if early_stop_criteria == "cer":
                    print("CRITERIA: CER")
                    if best_valid_val > avg_valid_cer:
                        count_stop = 0
                        best_valid_val = avg_valid_cer
                        save_model(model, vocab, (epoch+1), opt, metrics, args, best_model=True)
                    else:
                        print("count_stop:", count_stop)
                        count_stop += 1
                else:
                    print("CRITERIA: LOSS")
                    if best_valid_val > avg_valid_loss:
                        count_stop = 0
                        best_valid_val = avg_valid_loss
                        save_model(model, vocab, (epoch+1), opt, metrics, args, best_model=True)
                    else:
                        count_stop += 1
                        print("count_stop:", count_stop)

                if count_stop >= early_stop_val:
                    logging.info("EARLY STOP")
                    print("EARLY STOP\n")
                    break

            # if args.shuffle:
            #     logging.info("SHUFFLE")
            #     print("SHUFFLE\n")
