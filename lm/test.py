import argparse
import time
import math
import os
import unicodedata

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch import optim

import util.data as data
from model.rnn_model import *

import util.printhelper as printhelper
import util.masked_cross_entropy as masked_cross_entropy

from util.texthelper import is_contain_chinese_word

import sys

import numpy as np

parser = argparse.ArgumentParser(description='PyTorch SEAME RNN/LSTM Language Model')
parser.add_argument('--name', type=str, default='',
                    help='name')
parser.add_argument('--vocab-file-list', nargs='+', type=str)
parser.add_argument('--train_path', type=str, default='train.txt',
                    help='location of the data corpus')
parser.add_argument('--valid_path', type=str, default='valid.txt',
                    help='location of the data corpus')
parser.add_argument('--test_path', type=str, default='test.txt',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--pad', action='store_true',
                    help='pad the words')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log_path', type=str, default='./log', help='location of log file')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save_path', type=str,  default='save/',
                    help='path to save the final model')
args = parser.parse_args()

log_name = str(args.name) + "_model" + str(args.model) + "_bptt" + str(args.bptt) + "_lr" + str(args.lr) + "_drop" + str(args.dropout) + "_layers" + str(args.nlayers) + "_nhid" + str(args.nhid) + "_emsize" + str(args.emsize) + ".txt"
log_file = open(args.log_path + "/" + log_name, "w+")

save_path = args.save_path

dir_path = os.path.dirname(os.path.realpath(__file__))

is_pad = False
if args.pad:
    is_pad = args.pad

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

# Write all summary
printhelper.print_log(log_file, "is_pad\t:" + str(is_pad))
printhelper.print_log(log_file, "clip\t:" + str(args.clip))
printhelper.print_log(log_file, "start lr\t:" + str(args.lr))
printhelper.print_log(log_file, "em size\t:" + str(args.emsize))

def is_chinese_char(cc):
    return unicodedata.category(cc) == 'Lo'

def is_contain_chinese_word(seq):
    for i in range(len(seq)):
        if is_chinese_char(seq[i]):
            return True
    return False


###############################################################################
# Load data
###############################################################################

corpus = data.Corpus(args.train_path, args.valid_path, args.test_path, args.vocab_file_list, args.seed)

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    print(data.size(0), bsz)
    print("nbatch", nbatch)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data


eval_batch_size = 1

train_data = batchify(corpus.train, args.batch_size)
val_data = batchify(corpus.valid, eval_batch_size)
test_data = batchify(corpus.test, eval_batch_size)

val_data_batch10 = batchify(corpus.valid, 10)
test_data_batch10 = batchify(corpus.test, 10)

###############################################################################
# Build the model
###############################################################################

ntokens = len(corpus.dictionary)
model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

printhelper.print_log(log_file, str(model))
if args.cuda:
    model.cuda()

###############################################################################
# Training code
###############################################################################

# version 2.0
# def repackage_hidden(h):
#     """Wraps hidden states in new Variables, to detach them from their history."""
#     if type(h) == Variable:
#         return Variable(h.data)
#     else:
#         return tuple(repackage_hidden(v) for v in h)

# compatible to version 4.0
def repackage_hidden(h):
    """Wraps hidden states in new Tensors,
    to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

def get_batch(source, i, evaluation=False):
    seq_len = min(args.bptt, len(source) - 1 - i)
    if evaluation:
        with torch.no_grad():
            data = Variable(source[i:i+seq_len])
    else:
        data = Variable(source[i:i+seq_len])
    target = Variable(source[i+1:i+1+seq_len].view(-1))
    return data, target

# print the result
word2idx = corpus.dictionary.word2idx
idx2word = corpus.dictionary.idx2word

num_word = len(corpus.dictionary.idx2word)
english_word = {}
chinese_word = {}
for j in range(num_word):
    word = idx2word[j]
    if is_contain_chinese_word(word):
        chinese_word[j] = True
    else:
        english_word[j] = True

def evaluate(data_source, type_evaluation="val"):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(10)
    # hidden_lang = model.init_hidden(eval_batch_size)
    criterion = nn.CrossEntropyLoss()

    # if type_evaluation == "test":
    #     file_out = open("predictions/" + log_name, "w+", encoding="utf-8")

    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)

        output, hidden = model(data, hidden)

        if type_evaluation == "test":
            batch_size = output.size(1)
            seq_len = output.size(0)
            for k in range(batch_size):
                for j in range(seq_len-1):
                    en_word_val = 0
                    zh_word_val = 0

                    word_dist = output[j][k]
                    # for l in range(len(word_dist)):
                    #     if l in english_word:
                    #         en_word_val += pow(math.e, word_dist[l].data[0])
                    #     else:
                    #         zh_word_val += pow(math.e, word_dist[l].data[0])
                    # file_out.write(idx2word[data[j][k].data[0]] + "\t" + idx2word[data[j+1][k].data[0]] + "\t" + str(en_word_val) + "\t" + str(zh_word_val) + "\t" + str(en_val) + "\t" + str(zh_val) + "\n")

                    target_batch = targets.view(seq_len, batch_size)
                    target_word = target_batch[j][k].item()
                    word_val = pow(math.e, word_dist[target_word].item())
                    word_val_log = word_dist[target_word].item()
                    # word_dist, word_dist_idx = torch.topk(output[j][k], 1000, dim=-1)
                    # for l in range(len(word_dist)):
                    #     idx = word_dist_idx[l].data[0]
                    #     if idx in english_word:
                    #         en_word_val += pow(math.e, word_dist[l].data[0])
                    #     else:
                    #         zh_word_val += pow(math.e, word_dist[l].data[0])

                #     file_out.write(idx2word[data[j][k].item()] + "\t" + idx2word[data[j+1][k].item()] + "\t" + str(word_val) + "\t" + str(word_val_log) + "\n")
                # file_out.write("\n")

        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
        # hidden_lang = repackage_hidden(hidden_lang)
    return total_loss.item() / len(data_source)

def evaluate_test(data_source, type_evaluation="val"):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(eval_batch_size)
    # hidden_lang = model.init_hidden(eval_batch_size)
    criterion = nn.CrossEntropyLoss()

    total_en_en_loss, total_en_zh_loss, total_zh_en_loss, total_zh_zh_loss = 0, 0, 0, 0 # test
    total_data_en_en, total_data_en_zh, total_data_zh_en, total_data_zh_zh = 0, 0, 0, 0

    for i in range(0, data_source.size(0) - 1, args.bptt):
        data, targets = get_batch(data_source, i, evaluation=True)

        output, hidden = model(data, hidden)

        if type_evaluation == "test":
            # print("data>", data.squeeze().size())
            # print("targets>", targets.size())
            source_string, target_string = [], []
            source_lang, target_lang = [], []

            for i in range(len(data.squeeze())):
                word = corpus.dictionary.idx2word[data.squeeze()[i].item()]
                source_string.append(word)
                if is_contain_chinese_word(word):
                    source_lang.append(True)
                else:
                    source_lang.append(False)

            for i in range(len(targets)):
                word = corpus.dictionary.idx2word[targets[i].item()]
                target_string.append(word)
                if is_contain_chinese_word(word):
                    target_lang.append(True)
                else:
                    target_lang.append(False)

            # print("source:", source_string)
            # print("target", target_string)
            
            output_en_en, output_en_zh, output_zh_en, output_zh_zh = [], [], [], []
            target_en_en, target_en_zh, target_zh_en, target_zh_zh = [], [], [], []

            # print("output:", output.size())

            batch_size = output.size(1)
            seq_len = output.size(0)

            for i in range(len(source_lang)):
                source_word = corpus.dictionary.idx2word[data.squeeze()[i].item()]
                target_word = corpus.dictionary.idx2word[targets[i].item()]

                if source_word == "<eos>" or target_word == "<eos>":
                    # print("skip <eos>")
                    continue

                if source_lang[i] and target_lang[i]:
                    output_en_en.append(output[i])
                    target_en_en.append(targets[i])
                elif source_lang[i] and not target_lang[i]:
                    output_en_zh.append(output[i])
                    target_en_zh.append(targets[i])
                elif not source_lang[i] and target_lang[i]:
                    output_zh_en.append(output[i])
                    target_zh_en.append(targets[i])
                else:
                    output_zh_zh.append(output[i])
                    target_zh_zh.append(targets[i])

            total_data_en_en += len(output_en_en)
            total_data_en_zh += len(output_en_zh)
            total_data_zh_en += len(output_zh_en)
            total_data_zh_zh += len(output_zh_zh)

            if len(output_en_en) > 0:
                output_en_en = torch.stack(output_en_en)
                target_en_en = torch.LongTensor(target_en_en)
                if args.cuda:
                    output_en_en = output_en_en.cuda()
                    target_en_en = target_en_en.cuda()
                output_en_en_flat = output_en_en.view(-1, ntokens)
                total_en_en_loss += len(output_en_en) * criterion(output_en_en_flat, target_en_en).data
            if len(output_en_zh) > 0:
                output_en_zh = torch.stack(output_en_zh)
                target_en_zh = torch.LongTensor(target_en_zh)
                if args.cuda:
                    output_en_zh = output_en_zh.cuda()
                    target_en_zh = target_en_zh.cuda()
                output_en_zh_flat = output_en_zh.view(-1, ntokens)
                total_en_zh_loss += len(output_en_zh) * criterion(output_en_zh_flat, target_en_zh).data
            if len(output_zh_en) > 0:
                output_zh_en = torch.stack(output_zh_en)
                target_zh_en = torch.LongTensor(target_zh_en)
                if args.cuda:
                    output_zh_en = output_zh_en.cuda()
                    target_zh_en = target_zh_en.cuda()
                output_zh_en_flat = output_zh_en.view(-1, ntokens)
                total_zh_en_loss += len(output_zh_en) * criterion(output_zh_en_flat, target_zh_en).data
            if len(output_zh_zh) > 0:
                output_zh_zh = torch.stack(output_zh_zh)
                target_zh_zh = torch.LongTensor(target_zh_zh)
                if args.cuda:
                    output_zh_zh = output_zh_zh.cuda()
                    target_zh_zh = target_zh_zh.cuda()
                output_zh_zh_flat = output_zh_zh.view(-1, ntokens)
                total_zh_zh_loss += len(output_zh_zh) * criterion(output_zh_zh_flat, target_zh_zh).data

            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = repackage_hidden(hidden)
        else:
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).data
            hidden = repackage_hidden(hidden)
        # hidden_lang = repackage_hidden(hidden_lang)
    
    if type_evaluation == "test":
        print(total_data_en_en, total_data_en_zh, total_data_zh_en, total_data_zh_zh)
        print(total_en_en_loss.item(), total_en_zh_loss.item(), total_zh_en_loss.item(), total_zh_zh_loss.item())
        return total_loss.item() / len(data_source), total_en_en_loss.item() / total_data_en_en \
            , total_en_zh_loss.item() / total_data_en_zh, total_zh_en_loss.item() / total_data_zh_en \
                , total_zh_zh_loss.item() / total_data_zh_zh, (total_en_zh_loss.item() + total_zh_en_loss.item()) / (total_data_en_zh + total_data_zh_en)
    else:
        return total_loss.item() / len(data_source)

def train(data_source):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(corpus.dictionary)
    hidden = model.init_hidden(args.batch_size)
    hidden_lang = model.init_hidden(args.batch_size)
    criterion = nn.CrossEntropyLoss()
    
    batch_idx = 0

    num_batch = math.ceil(data_source.size(0) / args.bptt)
    # print(num_batch, data_source.size(0), args.bptt)
    indices = np.arange(num_batch)
    np.random.shuffle(indices)

    for batch, i in enumerate(range(0, data_source.size(0) - 1, args.bptt)):
        sys.stdout.flush()
        # print(">>", batch, i,  indices[batch] * args.bptt)
        data, targets = get_batch(data_source, indices[batch] * args.bptt)

        # Starting each batch, we detach the hidden state from how it was previously produced.
        # If we didn't, the model would try backpropagating all the way to start of the dataset.
        hidden = repackage_hidden(hidden)
        hidden_lang = repackage_hidden(hidden_lang)
        model.zero_grad()
        
        output, hidden = model(data, hidden)

        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        batch_idx += data.size(1)

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        opt = optim.SGD(model.parameters(), lr=lr)
        opt.step()

        total_loss += loss.data
        
        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss.item() / args.log_interval
            elapsed = time.time() - start_time

            log = '| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | word_loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(data_source) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss))

            printhelper.print_log(log_file, log)

            total_loss = 0
            start_time = time.time()

# Loop over epochs.
lr = args.lr
best_val_loss = None
counter = 0

# Load the best saved model.
with open(save_path, 'rb') as f:
    model = torch.load(f)

# Run on test data.
val_loss = evaluate(val_data_batch10, "dev")
log = ('=' * 89) + '| End of training | val loss {:5.2f} | val ppl {:8.2f}'.format(
    val_loss, math.exp(val_loss)) + ('=' * 89)
printhelper.print_log(log_file, log)

test_loss = evaluate(test_data_batch10, "dev")
log = ('=' * 89) + '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    val_loss, math.exp(test_loss)) + ('=' * 89)
printhelper.print_log(log_file, log)


test_loss, en_en_loss, en_zh_loss, zh_en_loss, zh_zh_loss, cs_loss = evaluate_test(val_data, "test")
log = ('=' * 89) + '| End of training | dev loss {:5.2f} | dev ppl {:8.2f} | en_en {:8.2f} | en_zh {:8.2f} | zh_en {:8.2f} | zh_zh {:8.2f} | cs_loss {:8.2f}'.format(
    test_loss, math.exp(test_loss), math.exp(en_en_loss), math.exp(en_zh_loss), math.exp(zh_en_loss), math.exp(zh_zh_loss), math.exp(cs_loss)) + ('=' * 89)
printhelper.print_log(log_file, log)

test_loss, en_en_loss, en_zh_loss, zh_en_loss, zh_zh_loss, cs_loss = evaluate_test(test_data, "test")
log = ('=' * 89) + '| End of training | test loss {:5.2f} | test ppl {:8.2f} | en_en {:8.2f} | en_zh {:8.2f} | zh_en {:8.2f} | zh_zh {:8.2f} | cs_loss {:8.2f}'.format(
    test_loss, math.exp(test_loss), math.exp(en_en_loss), math.exp(en_zh_loss), math.exp(zh_en_loss), math.exp(zh_zh_loss), math.exp(cs_loss)) + ('=' * 89)
printhelper.print_log(log_file, log)