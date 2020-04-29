import torch
import math
import numpy as np

from modules.common_layers import FactorizedMultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, FactorizedPositionwiseFeedForward, get_subsequent_mask, get_non_pad_mask, get_attn_key_pad_mask, get_attn_pad_mask, pad_list

def decode_hyp(vocab, hyp):
    """
    args: 
        hyp: list of hypothesis
    output:
        list of hypothesis (string)>
    """
    return "".join([vocab.id2label[int(x)] for x in hyp['yseq'][1:]])

def decode_greedy_search(decoder, vocab, encoder_padded_outputs, encoder_input_lengths, args, c_weight=1, start_token=-1):
    """
    Greedy search, decode 1-best utterance
    args:
        encoder_padded_outputs: B x T x H
    output:
        batch_ids_nbest_hyps: list of nbest in ids (size B)
        batch_strs_nbest_hyps: list of nbest in strings (size B)
    """
    ys = torch.ones(encoder_padded_outputs.size(0),1).fill_(start_token).long() # batch_size x 1
    if args.cuda:
        ys = ys.cuda()

    decoded_words = []
    for t in range(args.tgt_max_len - 1):
        pred_list, *_ = decoder(ys, encoder_padded_outputs, encoder_input_lengths) # B x T x V
        _, next_word = torch.max(pred_list[:, -1], dim=1)
        decoded_words.append([vocab.EOS_TOKEN if ni.item() == vocab.EOS_ID else vocab.id2label[ni.item()] for ni in next_word.view(-1)])
        next_word = next_word.unsqueeze(-1)
            
        if args.cuda:
            ys = torch.cat([ys, next_word.cuda()], dim=1)
            ys = ys.cuda()
        else:
            ys = torch.cat([ys, next_word], dim=1)

    sent = []
    for _, row in enumerate(np.transpose(decoded_words)):
        st = ''
        for e in row:
            if e == vocab.EOS_TOKEN:
                break
            else: 
                st += e
        sent.append(st)
    return sent

def decode_beam_search(decoder, vocab, encoder_padded_outputs, encoder_input_lengths, args, beam_width=2, beam_nbest=5, c_weight=1, start_token=-1):
    """
    Beam search, decode nbest utterances
    args:
        encoder_padded_outputs: B x T x H
        beam_size: int
        nbest: int
    output:
        batch_ids_nbest_hyps: list of nbest in ids (size B)
        batch_strs_nbest_hyps: list of nbest in strings (size B)
    """
    batch_size = encoder_padded_outputs.size(0)
    max_len = encoder_padded_outputs.size(1)

    batch_ids_nbest_hyps = []
    batch_strs_nbest_hyps = []

    for x in range(batch_size):
        encoder_output = encoder_padded_outputs[x].unsqueeze(0) # 1 x T x H

        # add SOS_TOKEN
        ys = torch.ones(1, 1).fill_(start_token).type_as(encoder_output).long()
        
        hyp = {'score': 0.0, 'yseq':ys}
        hyps = [hyp]
        ended_hyps = []

        # for i in range(args.tgt_max_len):
        for i in range(300):
            hyps_best_kept = []
            for j in range(len(hyps)):
                hyp = hyps[j]
                ys = hyp['yseq'] # 1 x i

                # print(encoder_padded_outputs[j].unsqueeze(0).size(), encoder_input_lengths[j].unsqueeze(0).size())
                pred_list, *_ = decoder(ys, encoder_padded_outputs[x].unsqueeze(0), encoder_input_lengths[x].unsqueeze(0)) # B x T x V
                # print(pred_list)
                # print(":.", pred_list.size())
                preds = pred_list.squeeze(0)
                # print(">", preds.size())
                local_best_scores, local_best_ids = torch.topk(preds, beam_width, dim=1)
                # print(local_best_scores.size())
                # print(hyps_best_kept)

                # calculate beam scores
                for j in range(beam_width):
                    new_hyp = {}
                    new_hyp["score"] = hyp["score"] + local_best_scores[-1, j]
                    new_hyp["yseq"] = torch.ones(1, (1+ys.size(1))).type_as(encoder_output).long()
                    new_hyp["yseq"][:, :ys.size(1)] = hyp["yseq"].cpu()
                    new_word = int(local_best_ids[-1, j])

                    # convert target index to source index
                    # print(j, new_word)
                    new_word = torch.LongTensor([new_word]).cuda()
                    new_hyp["yseq"][:, ys.size(1)] = new_word # adding new word
                    hyps_best_kept.append(new_hyp)
                hyps_best_kept = sorted(hyps_best_kept, key=lambda x:x["score"], reverse=True)[:beam_width]
            
            hyps = hyps_best_kept

            # add EOS_TOKEN
            if i == max_len - 1:
                for hyp in hyps:
                    hyp["yseq"] = torch.cat([hyp["yseq"], torch.ones(1,1).fill_(vocab.EOS_ID).type_as(encoder_output).long()], dim=1)

            # add hypothesis that have EOS_ID to ended_hyps list
            unended_hyps = []
            for hyp in hyps:
                if hyp["yseq"][0, -1] == vocab.EOS_ID:
                    seq_str = "".join(vocab.id2label[char.item()] for char in hyp["yseq"][0]).replace(vocab.PAD_TOKEN,"").replace(vocab.SOS_TOKEN,"").replace(vocab.EOS_TOKEN,"")
                    seq_str = seq_str.replace("  ", " ")
                    num_words = len(seq_str.split())
                    hyp["final_score"] = hyp["score"] + math.sqrt(num_words) * c_weight
                    ended_hyps.append(hyp)
                else:
                    unended_hyps.append(hyp)
            hyps = unended_hyps

            if len(hyps) == 0:
                # decoding process is finished
                break
            
        num_nbest = min(len(ended_hyps), beam_nbest)
        nbest_hyps = sorted(ended_hyps, key=lambda x:x["final_score"], reverse=True)[:num_nbest]
        a_nbest_hyps = sorted(ended_hyps, key=lambda x:x["final_score"], reverse=True)[:beam_width]

        sample_ids_nbest = []
        sample_strs_nbest = []
        for hyp in nbest_hyps:                
            hyp["yseq"] = hyp["yseq"][0].cpu().numpy().tolist()
            hyp_strs = decode_hyp(vocab, hyp)
            sample_ids_nbest.append(hyp["yseq"])
            sample_strs_nbest.append(hyp_strs)

        batch_ids_nbest_hyps.append(sample_ids_nbest[0])
        batch_strs_nbest_hyps.append(sample_strs_nbest[0])

    return batch_ids_nbest_hyps, batch_strs_nbest_hyps