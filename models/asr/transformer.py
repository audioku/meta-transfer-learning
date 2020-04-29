import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import numpy as np
import math
import os

from modules.decoding import decode_greedy_search, decode_beam_search
from modules.common_layers import FactorizedMultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, FactorizedPositionwiseFeedForward, get_subsequent_mask, get_non_pad_mask, get_attn_key_pad_mask, get_attn_pad_mask, pad_list
from torch.autograd import Variable
from utils.metrics import calculate_metrics

class Transformer(nn.Module):
    """
    Transformer class
    args:
        encoder: Encoder object
        decoder: Decoder object
    """

    def __init__(self, encoder, decoder, vocab, feat_extractor='vgg_cnn', train=True, is_factorized=False, r=100):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.vocab = vocab

        self.feat_extractor = feat_extractor
        self.is_factorized = is_factorized
        self.r = r
        
        self.copy_grad = None

        print("feat extractor:", feat_extractor)

        # feature embedding
        self.conv = None
        if feat_extractor == 'emb_cnn':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, kernel_size=(41, 11), stride=(2, 2), padding=(0, 10)),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True),
                nn.Conv2d(32, 32, kernel_size=(21, 11), stride=(2, 1), ),
                nn.BatchNorm2d(32),
                nn.Hardtanh(0, 20, inplace=True)
            )
        elif feat_extractor == 'vgg_cnn':
            self.conv = nn.Sequential(
                nn.Conv2d(1, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(64, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(128, 128, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )
        elif feat_extractor == "large_cnn":
            self.conv = nn.Sequential(
                nn.Conv2d(1, 32, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(32, 32, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2),
                nn.Conv2d(32, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.Conv2d(64, 64, 3, stride=1, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2, stride=2)
            )

        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)

    def encode(self, padded_input, input_lengths):
        """
        args:
            padded_input: B x 1 (channel for spectrogram=1) x (freq) x T
            padded_input: B x T x D
            input_lengths: B
        output:
            encoder_padded_outputs: B x T x H
        """
        # try:                
        if self.feat_extractor == 'emb_cnn' or self.feat_extractor == 'vgg_cnn' or self.feat_extractor == 'large_cnn':
            padded_input = self.conv(padded_input)

        # Reshaping features
        sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH
        
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)

        return encoder_padded_outputs

    def decode(self, encoder_padded_outputs, input_lengths, padded_target):
        """
        args:
            encoder_padded_outputs: B x T x H
            padded_input: B x T x D
            input_lengths: B
        output:
            pred: B x T x vocab
            gold: B x T
        """
        pred_list, gold_list, *_ = self.decoder(padded_target, encoder_padded_outputs, input_lengths)

        # hyp_list = []
        # print(pred_list.size())
        # print(gold_list.size())
        hyp_best_scores, hyp_best_ids = torch.topk(pred_list, 1, dim=2)
        hyp_list = hyp_best_ids.squeeze(2)
        # print(hyp_list.size())
        return pred_list, gold_list, hyp_list

    def forward(self, padded_input, input_lengths, padded_target, verbose=False):
        """
        args:
            padded_input: B x 1 (channel for spectrogram=1) x (freq) x T
            padded_input: B x T x D
            input_lengths: B
            padded_target: B x T
        output:
            pred: B x T x vocab
            gold: B x T
        """
        # try:                
        if self.feat_extractor == 'emb_cnn' or self.feat_extractor == 'vgg_cnn' or self.feat_extractor == 'large_cnn':
            padded_input = self.conv(padded_input)

        # Reshaping features
        sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH
        
        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        pred_list, gold_list, *_ = self.decoder(padded_target, encoder_padded_outputs, input_lengths)

        # hyp_list = []
        # print(pred_list.size())
        # print(gold_list.size())
        hyp_best_scores, hyp_best_ids = torch.topk(pred_list, 1, dim=2)
        hyp_list = hyp_best_ids.squeeze(2)
        # print(hyp_list.size())
        return pred_list, gold_list, hyp_list
        # except:
        #     torch.cuda.empty_cache()
    
    def decode_hyp(self, vocab, hyp):
        """
        args: 
            hyp: list of hypothesis
        output:
            list of hypothesis (string)>
        """
        return "".join([vocab.id2label[int(x)] for x in hyp['yseq'][1:]])

    def evaluate(self, padded_input, input_lengths, padded_target, args, beam_search=False, beam_width=0, beam_nbest=0, lm=None, lm_rescoring=False, lm_weight=0.1, c_weight=1, start_token=-1, verbose=False):
        """
        args:
            padded_input: B x T x D
            input_lengths: B
            padded_target: B x T
        output:
            batch_ids_nbest_hyps: list of nbest id
            batch_strs_nbest_hyps: list of nbest str
            batch_strs_gold: list of gold str
        """
        # TODO: Beam search only works with batch size of 1, need to be fix
        if self.feat_extractor == 'emb_cnn' or self.feat_extractor == 'vgg_cnn' or self.feat_extractor == 'large_cnn':
            padded_input = self.conv(padded_input)

        # Reshaping features
        sizes = padded_input.size() # B x H_1 (channel?) x H_2 x T
        padded_input = padded_input.view(sizes[0], sizes[1] * sizes[2], sizes[3])
        padded_input = padded_input.transpose(1, 2).contiguous()  # BxTxH

        encoder_padded_outputs, _ = self.encoder(padded_input, input_lengths)
        pred_list, gold_list, *_ = self.decoder(padded_target, encoder_padded_outputs, input_lengths)
        
        strs_gold = ["".join([self.vocab.id2label[int(x)] for x in gold_seq]) for gold_seq in gold_list]

        if beam_search:
            ids_hyps, strs_hyps = self.decoder.beam_search(encoder_padded_outputs, args, lm=lm, lm_rescoring=lm_rescoring, lm_weight=lm_weight, beam_width=args.beam_width, nbest=args.beam_nbest, c_weight=c_weight, start_token=start_token)
#             ids_hyps, strs_hyps = decode_beam_search(self.decoder, self.vocab, encoder_padded_outputs, input_lengths, args, beam_width=beam_width, beam_nbest=beam_nbest, c_weight=1, start_token=start_token)
            if len(strs_hyps) == 0:
                print(">>>>>>> switch to greedy")
                strs_hyps = self.decoder.greedy_search(encoder_padded_outputs, args, start_token=start_token)
#                 strs_hyps = decode_greedy_search(self.decoder, self.vocab, encoder_padded_outputs, input_lengths, args, c_weight=1, start_token=start_token)
            else:
                if len(strs_hyps[0].strip()) == 0:
                    print(">>>>>>> switch to greedy")
                    strs_hyps = self.decoder.greedy_search(encoder_padded_outputs, args, start_token=start_token)
        else:
            strs_hyps = self.decoder.greedy_search(encoder_padded_outputs, args, start_token=start_token)
#             strs_hyps = decode_greedy_search(self.decoder, self.vocab, encoder_padded_outputs, input_lengths, args, c_weight=1, start_token=start_token)

        return _, strs_hyps, strs_gold
               
    # Init copy_grad
    def init_copy_grad_(self):
        self.copy_grad = []
        for param in self.parameters():
            self.copy_grad.append(torch.zeros(param.shape, device=param.device, requires_grad=False))

    # Zeros copy_grad
    def zero_copy_grad(self):
        if self.copy_grad is None:
            self.init_copy_grad_()
        else:
            for i in range(len(self.copy_grad)):
                self.copy_grad[i] -= self.copy_grad[i]

    # Add model grad to copy_grad
    def add_copy_grad(self):
        if self.copy_grad is None:
            self.copy_grad = self.init_copy_grad_()

        for i, param in enumerate(self.parameters()):
            self.copy_grad[i].data += param.grad.data

    # Copy model grad to copy_grad
    def to_copy_grad(self):
        if self.copy_grad is None:
            self.copy_grad = self.init_copy_grad_()

        for i, param in enumerate(self.parameters()):
            self.copy_grad[i].data.copy_(param.grad)
            
    # Copy copy_grad to model grad
    def from_copy_grad(self):
        if self.copy_grad is None:
            self.copy_grad = self.init_copy_grad_()

        for i, param in enumerate(self.parameters()):
            param.grad.data.copy_(self.copy_grad[i])
