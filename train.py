import argparse
import json
import time
import math
import logging
import sys
import os
import torch
import torch.nn as nn
import numpy as np

from torchsummary import summary
from torch.autograd import Variable
from trainer.asr.trainer import Trainer
from utils.data import Vocab
from utils.data_loader import SpectrogramDataset, LogFBankDataset, AudioDataLoader, BucketingSampler
from utils.functions import save_model, load_model, init_transformer_model, init_optimizer, compute_num_params, generate_labels

parser = argparse.ArgumentParser(description='Transformer ASR training')
parser.add_argument('--model', default='TRFS', type=str, help="")
parser.add_argument('--name', default='model', help="Name of the model for saving")

parser.add_argument('--train-manifest-list', nargs='+', type=str)
parser.add_argument('--valid-manifest-list', nargs='+', type=str)
parser.add_argument('--test-manifest-list', nargs='+', type=str)

parser.add_argument('--sample-rate', default=22050, type=int, help='Sample rate')
parser.add_argument('--batch-size', default=20, type=int, help='Batch size for training')
parser.add_argument('--num-workers', default=4, type=int, help='Number of workers used in data-loading')
parser.add_argument('--labels-path', default='labels.json', help='Contains all characters for transcription')
parser.add_argument('--label-smoothing', default=0.0, type=float, help='Label smoothing')
parser.add_argument('--window-size', default=.02, type=float, help='Window size for spectrogram in seconds')
parser.add_argument('--window-stride', default=.01, type=float, help='Window stride for spectrogram in seconds')
parser.add_argument('--window', default='hamming', help='Window type for spectrogram generation')
parser.add_argument('--epochs', default=1000, type=int, help='Number of training epochs')
parser.add_argument('--cuda', dest='cuda', action='store_true', help='Use cuda to train model')
# parser.add_argument('--lr', '--learning-rate', default=3e-4, type=float, help='initial learning rate')
parser.add_argument('--early-stop', default="loss,10", type=str, help='Early stop (loss,10) or (cer,10)')
parser.add_argument('--save-every', default=5, type=int, help='Save model every certain number of epochs')
parser.add_argument('--save-folder', default='models/', help='Location to save epoch models')
parser.add_argument('--emb-trg-sharing', action='store_true', help='Share embedding weight source and target')
parser.add_argument('--feat_extractor', default='vgg_cnn', type=str, help='emb_cnn or vgg_cnn or none')
parser.add_argument('--feat', type=str, default='spectrogram', help='spectrogram or logfbank')
parser.add_argument('--verbose', action='store_true', help='Verbose')

parser.add_argument('--continue-from', default='', type=str, help='Continue from checkpoint model')
parser.add_argument('--augment', dest='augment', action='store_true', help='Use random tempo and gain perturbations.')
parser.add_argument('--noise-dir', default=None,
                    help='Directory to inject noise into audio. If default, noise Inject not added')
parser.add_argument('--noise-prob', default=0.4, help='Probability of noise being added per sample')
parser.add_argument('--noise-min', default=0.0,
                    help='Minimum noise level to sample from. (1.0 means all noise, not original signal)', type=float)
parser.add_argument('--noise-max', default=0.5,
                    help='Maximum noise levels to sample from. Maximum 1.0', type=float)

# Transformer
parser.add_argument('--num-enc-layers', default=3, type=int, help='Number of layers')
parser.add_argument('--num-dec-layers', default=3, type=int, help='Number of layers')
parser.add_argument('--num-heads', default=5, type=int, help='Number of heads')
parser.add_argument('--dim-model', default=512, type=int, help='Model dimension')
parser.add_argument('--dim-key', default=64, type=int, help='Key dimension')
parser.add_argument('--dim-value', default=64, type=int, help='Value dimension')
parser.add_argument('--dim-input', default=161, type=int, help='Input dimension')
parser.add_argument('--dim-inner', default=1024, type=int, help='Inner dimension')
parser.add_argument('--dim-emb', default=512, type=int, help='Embedding dimension')

parser.add_argument('--src-max-len', default=2500, type=int, help='Source max length')
parser.add_argument('--tgt-max-len', default=1000, type=int, help='Target max length')

# Noam optimizer
parser.add_argument('--warmup', default=4000, type=int, help='Warmup')
parser.add_argument('--min-lr', default=1e-5, type=float, help='min lr')
parser.add_argument('--k-lr', default=1, type=float, help='factor lr')

# SGD optimizer
parser.add_argument('--lr', default=1e-4, type=float, help='lr')
parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
parser.add_argument('--lr-anneal', default=1.1, type=float, help='lr anneal')

# Decoder search
parser.add_argument('--beam-search', action='store_true', help='Beam search')
parser.add_argument('--beam-width', default=3, type=int, help='Beam size')
parser.add_argument('--beam-nbest', default=5, type=int, help='Number of best sequences')
parser.add_argument('--lm-rescoring', action='store_true', help='Rescore using LM')
parser.add_argument('--lm-path', type=str, default="lm_model.pt", help="Path to LM model")
parser.add_argument('--lm-weight', default=0.1, type=float, help='LM weight')
parser.add_argument('--c-weight', default=0.1, type=float, help='Word count weight')
parser.add_argument('--prob-weight', default=1.0, type=float, help='Probability E2E weight')

# loss
parser.add_argument('--loss', type=str, default='ce', help='ce or ctc')
parser.add_argument('--clip', action='store_true', help="clip")
parser.add_argument('--max-norm', default=400, type=float, help="max norm for clipping")
parser.add_argument('--is-accu-loss', action='store_true', help="is accu loss. experimental")
parser.add_argument('--is-factorized', action='store_true', help="is factorized. experimental")
parser.add_argument('--r', default=100, type=int, help='rank')
parser.add_argument('--dropout', default=0.1, type=float, help='Dropout')

# shuffle
parser.add_argument('--shuffle', action='store_true', help='Shuffle')

# input
parser.add_argument('--input_type', type=str, default='char', help='char or bpe or ipa')

# Post-training factorization
parser.add_argument('--rank', default=10, type=float, help="rank")
parser.add_argument('--factorize', action='store_true', help='factorize')

torch.manual_seed(123456)
torch.cuda.manual_seed_all(123456)

args = parser.parse_args()
USE_CUDA = args.cuda

if __name__ == '__main__':
    print("="*50)
    print("THE EXPERIMENT LOG IS SAVED IN: " + "log/" + args.name)
    print("TRAINING MANIFEST: ", args.train_manifest_list)
    print("VALID MANIFEST: ", args.valid_manifest_list)
    print("TEST MANIFEST: ", args.test_manifest_list)
    print("INPUT TYPE: ", args.input_type)
    print("="*50)

    if not os.path.exists("./log"): os.mkdir("./log")
    for handler in logging.root.handlers[:]: logging.root.removeHandler(handler)
        
    if args.continue_from == '':
        logging.basicConfig(filename="log/" + args.name + ".log", filemode='w+', format='%(asctime)s - %(message)s', level=logging.INFO)
        print("TRAINING FROM SCRATCH")
        logging.info("TRAINING FROM SCRATCH")
    else:
        logging.basicConfig(filename="log/" + args.name + ".log", filemode='a+', format='%(asctime)s - %(message)s', level=logging.INFO)
        print("RESUME TRAINING")
        logging.info("RESUME TRAINING")

    audio_conf = dict(sample_rate=args.sample_rate,
                      window_size=args.window_size,
                      window_stride=args.window_stride,
                      window=args.window,
                      noise_dir=args.noise_dir,
                      noise_prob=args.noise_prob,
                      noise_levels=(args.noise_min, args.noise_max))

    logging.info(audio_conf)
    
    with open(args.labels_path, encoding="utf-8") as label_file:
        labels = json.load(label_file)

    vocab = Vocab()
    for label in labels:
        vocab.add_token(label)
        vocab.add_label(label)
        
    if args.feat == "spectrogram":

        train_data = SpectrogramDataset(vocab, args, audio_conf, manifest_filepath_list=args.train_manifest_list, normalize=True, augment=args.augment, input_type=args.input_type, is_train=True)
    elif args.feat == "logfbank":
        train_data = LogFBankDataset(vocab, args, audio_conf, manifest_filepath_list=args.train_manifest_list, normalize=True, augment=args.augment, input_type=args.input_type, is_train=True)
    train_sampler = BucketingSampler(train_data, batch_size=args.batch_size)
    train_loader = AudioDataLoader(pad_token_id=0, dataset=train_data, num_workers=args.num_workers, batch_sampler=train_sampler)

    valid_loader_list, test_loader_list = [], []
    for i in range(len(args.valid_manifest_list)):
        if args.feat == "spectrogram":
            valid_data = SpectrogramDataset(vocab, args, audio_conf, manifest_filepath_list=[args.valid_manifest_list[i]], normalize=True, augment=args.augment, input_type=args.input_type)
        elif args.feat == "logfbank":
            valid_data = LogFBankDataset(vocab, args, audio_conf, manifest_filepath_list=[args.valid_manifest_list[i]], normalize=True, augment=False, input_type=args.input_type)
        valid_sampler = BucketingSampler(valid_data, batch_size=args.batch_size)
        valid_loader = AudioDataLoader(pad_token_id=0, dataset=valid_data, num_workers=args.num_workers)
        valid_loader_list.append(valid_loader)

    start_epoch = 0
    metrics = None
    loaded_args = None
    if args.continue_from != "":
        logging.info("Continue from checkpoint:" + args.continue_from)
        model, vocab, opt, epoch, metrics, loaded_args = load_model(args.continue_from)
        start_epoch = (epoch)  # index starts from zero
        verbose = args.verbose
    else:
        if args.model == "TRFS":
            model = init_transformer_model(args, vocab, is_factorized=args.is_factorized, r=args.r)
            opt = init_optimizer(args, model, "noam")
        else:
            logging.info("The model is not supported, check args --h")
    
    loss_type = args.loss

    if USE_CUDA:
        model = model.cuda()

    logging.info(model)
    num_epochs = args.epochs

    print("Parameters: {}(trainable), {}(non-trainable)".format(compute_num_params(model)[0], compute_num_params(model)[1]))

    trainer = Trainer()
    trainer.train(model, vocab, train_loader, train_sampler, valid_loader_list, opt, loss_type, start_epoch, num_epochs, args, metrics, early_stop=args.early_stop)
