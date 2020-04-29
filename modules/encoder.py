import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I
import numpy as np
import math
import os

from .common_layers import FactorizedMultiHeadAttention, PositionalEncoding, PositionwiseFeedForward, FactorizedPositionwiseFeedForward, get_subsequent_mask, get_non_pad_mask, get_attn_key_pad_mask, get_attn_pad_mask, pad_list_with_mask

from torch.autograd import Variable
# from utils import constant
from utils.metrics import calculate_metrics

class Encoder(nn.Module):
    """ 
    Encoder Transformer class
    """

    def __init__(self, num_layers, num_heads, dim_model, dim_key, dim_value, dim_input, dim_inner, dropout=0.1, src_max_length=2500, is_factorized=False, r=100):
        super(Encoder, self).__init__()

        self.dim_input = dim_input
        self.num_layers = num_layers
        self.num_heads = num_heads

        self.dim_model = dim_model
        self.dim_key = dim_key
        self.dim_value = dim_value
        self.dim_inner = dim_inner

        self.src_max_length = src_max_length

        self.is_factorized = is_factorized
        self.r = r

        self.dropout = nn.Dropout(dropout)
        self.dropout_rate = dropout

        if is_factorized:
            self.input_linear_a = nn.Linear(dim_input, r, bias=False)
            self.input_linear_b = nn.Linear(r, dim_model)
        else:
            self.input_linear = nn.Linear(dim_input, dim_model)
        self.layer_norm_input = nn.LayerNorm(dim_model)
        self.positional_encoding = PositionalEncoding(
            dim_model, src_max_length)

        self.layers = nn.ModuleList([
            EncoderLayer(num_heads, dim_model, dim_inner, dim_key, dim_value, dropout=dropout, is_factorized=is_factorized, r=r) for _ in range(num_layers)
        ])

    def forward(self, padded_input, input_lengths):
        """
        args:
            padded_input: B x T x D
            input_lengths: B
        return:
            output: B x T x H
        """
        encoder_self_attn_list = []

        # Prepare masks
        non_pad_mask = get_non_pad_mask(padded_input, input_lengths=input_lengths)  # B x T x D
        seq_len = padded_input.size(1)
        self_attn_mask = get_attn_pad_mask(padded_input, input_lengths, seq_len)  # B x T x T

        if self.is_factorized:
            encoder_output = self.layer_norm_input(self.input_linear_b(self.input_linear_a(
                padded_input))) + self.positional_encoding(padded_input)
        else:
            encoder_output = self.layer_norm_input(self.input_linear(
                padded_input)) + self.positional_encoding(padded_input)
        
        for layer in self.layers:
            encoder_output, self_attn = layer(
                encoder_output, non_pad_mask=non_pad_mask, self_attn_mask=self_attn_mask)
            encoder_self_attn_list += [self_attn]

        return encoder_output, encoder_self_attn_list


class EncoderLayer(nn.Module):
    """
    Encoder Layer Transformer class
    """

    def __init__(self, num_heads, dim_model, dim_inner, dim_key, dim_value, dropout=0.1, is_factorized=False, r=100):
        super(EncoderLayer, self).__init__()
        self.is_factorized = is_factorized
        self.r = r
        self.self_attn = FactorizedMultiHeadAttention(num_heads, dim_model, dim_key, dim_value, dropout=dropout, r=r)
        if is_factorized:
            self.pos_ffn = FactorizedPositionwiseFeedForward(dim_model, dim_inner, dropout=dropout, r=r)
        else:
            self.pos_ffn = PositionwiseFeedForward(dim_model, dim_inner, dropout=dropout)

    def forward(self, enc_input, non_pad_mask=None, self_attn_mask=None):
        enc_output, self_attn = self.self_attn(
            enc_input, enc_input, enc_input, mask=self_attn_mask)
        enc_output *= non_pad_mask

        enc_output = self.pos_ffn(enc_output)
        enc_output *= non_pad_mask

        return enc_output, self_attn
