import os
import torch
from torch.autograd import Variable
import math

import util.datahelper as datahelper
import util.texthelper as texthelper

import random
import numpy as np

class LMDataset(object):
    def __init__(self, task_list, args):
        self.bptt = args.bptt
        self.batch_size = args.batch_size
        
        self.args = args

        self.task_list = []
        for i in range(len(task_list)):
            self.task_list.append(self.batchify(task_list[i], self.batch_size))

        super(LMDataset, self).__init__()

    def batchify(self, data, bsz):
        # Work out how cleanly we can divide the dataset into bsz parts.
        nbatch = data.size(0) // bsz
        # Trim off any extra elements that wouldn't cleanly fit (remainders).
        data = data.narrow(0, 0, nbatch * bsz)
        # Evenly divide the data across the bsz batches.
        data = data.view(bsz, -1).t().contiguous()
        if self.args.cuda:
            data = data.cuda()
        return data

    def get_batch(self, source, i, evaluation=False):
        seq_len = min(self.bptt, len(source) - 1 - i)
        if evaluation:
            with torch.no_grad():
                data = Variable(source[i:i+seq_len])
        else:
            data = Variable(source[i:i+seq_len])
        target = Variable(source[i+1:i+1+seq_len].view(-1))
        return data, target

    def sample(self, manifest_id, i):
        def func(p):
            return p.size(1)

        def func_trg(p):
            return len(p)

        ids = self.task_list[manifest_id]
        num_batch = math.ceil(ids.size(0) / self.bptt)
        # shuffled_indices = np.random.choice(np.arange(0, num_batch), 2, replace=True)
        # tr_id = shuffled_indices[0]
        # val_id = shuffled_indices[1]
        tr_id = i
        val_id = i+1
        
        tr_ids = ((tr_id * self.bptt) % len(ids)) - (((tr_id * self.bptt) % len(ids)) % self.bptt)
        val_ids = (((val_id) * self.bptt) % len(ids)) - ((((val_id) * self.bptt) % len(ids)) % self.bptt)

        tr_src, tr_target = self.get_batch(ids, tr_ids)
        val_src, val_target = self.get_batch(ids, val_ids)           

        return (tr_src, tr_target, val_src, val_target)

class Dictionary(object):
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word[len(self.idx2word)] = word
            self.word2idx[word] = len(self.idx2word) - 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word)

class Corpus(object):
    def __init__(self, train_path, valid_path=None, test_path=None, dictionary=None, seed=1000):
        random.seed(seed)
        if dictionary is None:
            self.dictionary = Dictionary()
        else:
            self.dictionary = dictionary
            print("load dictionary")

        self.train, self.train_lang = self.tokenize(train_path, True)
        print("train:", len(self.dictionary))

        if valid_path is not None:
            self.valid, self.valid_lang = self.tokenize(valid_path, False)
            print("valid:", len(self.dictionary))
        else:
            print("valid_path is None")

        if test_path is not None:
            self.test, self.test_lang = self.tokenize(test_path, False)
            print("test:", len(self.dictionary))
        else:
            print("test_path is None")

        print("dictionary size:", len(self.dictionary))

    def create_seq_idx_matrix(self, path):
        assert os.path.exists(path)

        matrix = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.replace("  ", " ")
                words = line.split() + ['<eos>']
                word_tokens = torch.LongTensor(len(words))
                for i in range(len(words)):
                    word = words[i]

                    if not word in self.dictionary.word2idx:
                        word_id = self.dictionary.word2idx["<oov>"]
                    else:
                        word_id = self.dictionary.word2idx[word]
                        
                    word_tokens[i] = word_id
                matrix.append(word_tokens.unsqueeze_(1))
        return matrix

    def create_seq_word_matrix(self, path):
        assert os.path.exists(path)

        matrix = []
        with open(path, 'r') as f:
            for line in f:
                line = line.strip()
                line = line.replace("  ", " ")
                words = line.split() + ['<eos>']
                word_tokens = []
                for word in words:
                    word_tokens.append(word)
                matrix.append(word_tokens)
        return matrix

    def tokenize(self, path, save, randomize=False):
        """Tokenizes a text file."""
        assert os.path.exists(path)

        # Add words to the dictionary
        self.dictionary.add_word("<oov>")

        data = []
        with open(path, 'r') as f:
            for line in f:
                data.append(line)

        # if randomize:
        #     random.shuffle(data)

        # with open(path, 'r') as f:
        tokens = 0
        for i in range(len(data)):
            line = data[i]
            line = line.strip().lower()
            line = line.replace("  ", " ")
            words = line.split() + ['<eos>']
            tokens += len(words)
            for word in words:
                if save:
                    self.dictionary.add_word(word)

        # Tokenize file content
        with open(path, 'r') as f:
            ids = torch.LongTensor(tokens)
            langs = torch.LongTensor(tokens)
            token = 0
            for line in f:
                line = line.strip().lower()
                line = line.replace("  ", " ")
                words = line.split() + ['<eos>']
                for word in words:
                    if not word in self.dictionary.word2idx:
                        ids[token] = self.dictionary.word2idx["<oov>"]
                    else:
                        ids[token] = self.dictionary.word2idx[word]

                    if texthelper.is_contain_chinese_word(word):
                        langs[token] = 1
                    else:
                        langs[token] = 0

                    token += 1

        return ids, langs