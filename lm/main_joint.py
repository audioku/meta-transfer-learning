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

# import util.printhelper as printhelper
import util.masked_cross_entropy as masked_cross_entropy

from copy import deepcopy

import sys

parser = argparse.ArgumentParser(description='PyTorch SEAME RNN/LSTM Language Model')
parser.add_argument('--name', type=str, default='',
                    help='name')
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
parser.add_argument('--ratio', type=float, default=0.8,
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
parser.add_argument('--save', type=str,  default='./model',
                    help='path to save the final model')
args = parser.parse_args()

log_name = str(args.name) + "_model" + str(args.model) + "_bptt" + str(args.bptt) + "_lr" + str(args.lr) + "_drop" + str(args.dropout) + "_layers" + str(args.nlayers) + "_nhid" + str(args.nhid) + "_emsize" + str(args.emsize) + "_ratio" + str(args.ratio) + ".txt"
log_file = open(args.log_path + "/" + log_name, "w+")

save_path = args.save + "/" + log_name + ".pt"

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
print(log_file, "is_pad\t:" + str(is_pad))
print(log_file, "clip\t:" + str(args.clip))
print(log_file, "start lr\t:" + str(args.lr))
print(log_file, "em size\t:" + str(args.emsize))

def is_chinese_char(cc):
    return unicodedata.category(cc) == 'Lo'

def is_contain_chinese_word(seq):
    for i in range(len(seq)):
        if is_chinese_char(seq[i]):
            return True
    return False

def batchify(data, bsz):
    # Work out how cleanly we can divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # print(data.size(0), bsz)
    # print("nbatch", nbatch)
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    if args.cuda:
        data = data.cuda()
    return data

###############################################################################
# Load data
###############################################################################
eval_batch_size = 10

dictionary = data.Dictionary()

# data path
SEAME_TRAIN_PATH = "./data/seame_train.txt"
SEAME_VALID_PATH = "./data/seame_valid.txt"
SEAME_TEST_PATH = "./data/seame_test.txt"

CV_TRAIN_PATH = "./data/cv_train.txt"
CV_VALID_PATH = "./data/cv_valid.txt"
CV_TEST_PATH = "./data/cv_test.txt"

HKUST_TRAIN_PATH = "./data/hkust_train.txt"
HKUST_TEST_PATH = "./data/hkust_dev.txt"

# corpus
print("> Reading SEAME data")
seame_corpus = data.Corpus(SEAME_TRAIN_PATH, SEAME_VALID_PATH, SEAME_TEST_PATH, None, args.seed)
print("vocab:", len(seame_corpus.dictionary))
print("> Reading CV data")
cv_corpus = data.Corpus(CV_TRAIN_PATH, CV_VALID_PATH, CV_TEST_PATH, seame_corpus.dictionary, args.seed)
print("vocab:", len(cv_corpus.dictionary))
print("> Reading HKUST data")
hkust_corpus = data.Corpus(HKUST_TRAIN_PATH, None, HKUST_TEST_PATH, cv_corpus.dictionary, args.seed)
print("vocab:", len(hkust_corpus.dictionary))
dictionary = hkust_corpus.dictionary

# train data
print("> Preparing training data")
train_datasets = [cv_corpus.train, hkust_corpus.train, seame_corpus.train]
lm_dataset = data.LMDataset(train_datasets, args)

# val data
print("> Preparing valid data")
seame_val_data = batchify(seame_corpus.valid, eval_batch_size)
cv_val_data = batchify(cv_corpus.valid, eval_batch_size)

# test data
print("> Preparing test data")
seame_test_data = batchify(seame_corpus.test, eval_batch_size)
cv_test_data = batchify(cv_corpus.test, eval_batch_size)
hkust_test_data = batchify(hkust_corpus.test, eval_batch_size)

###############################################################################
# Build the model
###############################################################################

ntokens = len(dictionary)
model = RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)

print(model)
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
word2idx = dictionary.word2idx
idx2word = dictionary.idx2word

num_word = len(dictionary.idx2word)
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
    ntokens = len(dictionary)
    hidden = model.init_hidden(eval_batch_size)
    # hidden_lang = model.init_hidden(eval_batch_size)
    criterion = nn.CrossEntropyLoss()

    if "test" in type_evaluation:
        name = type_evaluation.split("_")
        file_out = open("predictions/" + name + "_" + log_name, "w+", encoding="utf-8")

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

                    file_out.write(idx2word[data[j][k].item()] + "\t" + idx2word[data[j+1][k].item()] + "\t" + str(word_val) + "\t" + str(word_val_log) + "\n")
                file_out.write("\n")

        output_flat = output.view(-1, ntokens)
        total_loss += len(data) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
        # hidden_lang = repackage_hidden(hidden_lang)
    return total_loss.item() / len(data_source)

def forward_one_batch(model, hidden, inputs):
    # Starting each batch, we detach the hidden state from how it was previously produced.
    # If we didn't, the model would try backpropagating all the way to start of the dataset.
    hidden = repackage_hidden(hidden)
    model.zero_grad()
    output, hidden = model(inputs, hidden)
    return output, hidden

def train(tr_data_source, val_data_source, start_it, num_it, log_interval, valid_interval):
    # Turn on training mode which enables dropout.
    model.train()
    total_loss = 0
    start_time = time.time()
    ntokens = len(dictionary)
    hidden = model.init_hidden(args.batch_size)
    criterion = nn.CrossEntropyLoss()

    weights_original = None

    lr = args.lr
    it = start_it
    total_loss = 0
    best_val_loss = 0
    while it < num_it:
        batch_loss = 0
        outer_opt = optim.SGD(model.parameters(), lr=lr)

        if it % 100 == 0:
            print(">", it)

        weights_original = deepcopy(model.state_dict())
        # hidden = repackage_hidden(hidden)

        _, _, val_inputs, val_targets = tr_data_source.sample(-1, it)
        if args.cuda:
            val_inputs = val_inputs.cuda()
            val_targets = val_targets.cuda()
    
        # print(tr_inputs.size(), tr_targets.size())

        for i in range(len(tr_data_source.task_list)):
            sys.stdout.flush()
            tr_inputs, tr_targets, _, _ = tr_data_source.sample(i, it)

            if args.cuda:
                tr_inputs = tr_inputs.cuda()
                tr_targets = tr_targets.cuda()

            # Meta Train
            model.train()
        
            output, hidden = forward_one_batch(model, hidden, tr_inputs)

            tr_loss = criterion(output.view(-1, ntokens), tr_targets)

            if i < 2:
                tr_loss = tr_loss * (1-args.ratio)/2
            else:
                tr_loss = tr_loss * args.ratio

            batch_loss += tr_loss

            # Inner Backward
            tr_loss.backward()
        
        total_loss += batch_loss

        if args.clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        outer_opt.step()

        if it % log_interval == 0 and it > 0:
            # cur_loss = total_loss.item() / args.log_interval
            if (it % valid_interval) == 0:
                cur_loss = total_loss.item() / valid_interval
            else:
                cur_loss = total_loss.item() / (it % valid_interval)
            elapsed = time.time() - start_time

            log = '| it {:3d} | lr {:02.2f} | ms/batch {:5.2f} | word_loss {:5.2f} | avg ppl {:8.2f}'.format(
                it, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss))

            # printhelper.print_log(log_file, log)
            print(log)

            start_time = time.time()

        # validation
        if it % valid_interval == 0 and it > 0:
            val_loss = evaluate(val_data_source)

            print("it {} | val loss {:5f} | ppl {:5f}".format(it, val_loss, math.exp(val_loss)))
            
            test_loss = evaluate(seame_test_data)
            print("it {} | test loss {:5f} | ppl {:5f}".format(it, test_loss, math.exp(test_loss)))

            
            # Save the model if the validation loss is the best we've seen so far.
            if not best_val_loss or val_loss < best_val_loss:
                with open(save_path, 'wb') as f:
                    torch.save(model, f)
                best_val_loss = val_loss
                counter = 0
            else:
                lr /= 4.0
                counter += 1

            if counter == 5:
                break

            total_loss = 0
        it += 1

# Loop over epochs.
lr = args.lr
best_val_loss = None
counter = 0
log_interval = 200
valid_interval = 600

# At any point you can hit Ctrl + C to break out of training early.
try:
    print("############# TRAIN data #############")
    train(lm_dataset, seame_val_data, 0, 1000000, log_interval, valid_interval)

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

# Load the best saved model.
with open(save_path, 'rb') as f:
    model = torch.load(f)

# Run on test data.
seame_test_loss = evaluate(seame_test_data)
cv_test_loss = evaluate(cv_test_data)
hkust_test_loss = evaluate(hkust_test_data)

log = 'SEAME ' + ('=' * 89) + '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    seame_test_loss, math.exp(seame_test_loss)) + ('=' * 89)
print(log_file, log)

log = 'CV ' + ('=' * 89) + '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    cv_test_loss, math.exp(cv_test_loss)) + ('=' * 89)
print(log_file, log)

log = 'HKUST ' + ('=' * 89) + '| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    hkust_test_loss, math.exp(hkust_test_loss)) + ('=' * 89)
print(log_file, log)
# log_file.close()