import torch
import os
import math
import torch.nn as nn
import logging
import numpy as np

# from modules import CPT2LMHeadModel
# from models.asr.transformer_cpt2 import TransformerCPT2, Encoder, Decoder
from models.asr.transformer import Transformer
from transformers import BertModel
from modules import Encoder, Decoder, Discriminator
from models.asr.transformer import Transformer
from utils.optimizer import NoamOpt, AnnealingOpt

def generate_labels(labels, special_token_list):
    # add PAD_CHAR, SOS_CHAR, EOS_CHAR, UNK_CHAR
    label2id, id2label = {}, {}
    count = 0

    for i in range(len(special_token_list)):
        label2id[special_token_list[i]] = count
        id2label[count] = special_token_list[i]
        count += 1
    
    for i in range(len(labels)):
        if labels[i] not in label2id:
            labels[i] = labels[i]
            label2id[labels[i]] = count
            id2label[count] = labels[i]
            count += 1
        else:
            print("multiple label: ", labels[i])
    return label2id, id2label

def compute_num_params(model):
    """
    Computes number of trainable and non-trainable parameters
    """
    sizes = [(np.array(p.data.size()).prod(), int(p.requires_grad)) for p in model.parameters()]
    return sum(map(lambda t: t[0]*t[1], sizes)), sum(map(lambda t: t[0]*(1 - t[1]), sizes))

def save_joint_model(model, vocab, epoch, opt, metrics, args, best_model=False):
    """
    Saving model, TODO adding history
    """
    if best_model:
        save_path = "{}/{}/best_model.th".format(
            args.save_folder, args.name)
    else:
        save_path = "{}/{}/epoch_{}.th".format(args.save_folder,
                                               args.name, epoch)

    if not os.path.exists(args.save_folder + "/" + args.name):
        os.makedirs(args.save_folder + "/" + args.name)

    print("SAVE MODEL to", save_path)
    logging.info("SAVE MODEL to " + save_path)
    if args.loss == "ce":
        args = {
            'vocab': vocab,
            'args': args,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt': opt,
            'metrics': metrics
        }
    else:
        print("Loss is not defined")
        logging.info("Loss is not defined")
    torch.save(args, save_path)

def save_discriminator(discriminator, epoch, opt, args, best_model=False):
    """
    Saving discriminator
    """
    if best_model:
        save_path = "{}/{}/best_discriminator.th".format(
            args.save_folder, args.name)
    else:
        save_path = "{}/{}/epoch_{}.th".format(args.save_folder,
                                               args.name, epoch)

    if not os.path.exists(args.save_folder + "/" + args.name):
        os.makedirs(args.save_folder + "/" + args.name)
    
    print("SAVE DISCRIMINATOR to", save_path)
    logging.info("SAVE DISCRIMINATOR to " + save_path)
    if args.loss == "ce":
        args = {
            'args': args,
            'epoch': epoch,
            'model_state_dict': discriminator.state_dict(),
            'opt': opt
        }
    else:
        print("Loss is not defined")
        logging.info("Loss is not defined")
    torch.save(args, save_path)

def save_meta_model(model, vocab, epoch, inner_opt, outer_opt, metrics, args, best_model=False):
    """
    Saving model, TODO adding history
    """
    if best_model:
        save_path = "{}/{}/best_model.th".format(
            args.save_folder, args.name)
    else:
        save_path = "{}/{}/epoch_{}.th".format(args.save_folder,
                                               args.name, epoch)

    if not os.path.exists(args.save_folder + "/" + args.name):
        os.makedirs(args.save_folder + "/" + args.name)

    print("SAVE MODEL to", save_path)
    logging.info("SAVE MODEL to " + save_path)
    args = {
        'vocab': vocab,
        'args': args,
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'inner_opt': inner_opt,
        'outer_opt': outer_opt,
        'metrics': metrics
    }
    torch.save(args, save_path)

def save_model(model, vocab, epoch, opt, metrics, args, best_model=False):
    """
    Saving model, TODO adding history
    """
    if best_model:
        save_path = "{}/{}/best_model.th".format(
            args.save_folder, args.name)
    else:
        save_path = "{}/{}/epoch_{}.th".format(args.save_folder,
                                               args.name, epoch)

    if not os.path.exists(args.save_folder + "/" + args.name):
        os.makedirs(args.save_folder + "/" + args.name)

    print("SAVE MODEL to", save_path)
    logging.info("SAVE MODEL to " + save_path)
    if args.loss == "ce":
        args = {
            'vocab': vocab,
            'args': args,
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'opt': opt,
            'metrics': metrics
        }
    else:
        print("Loss is not defined")
        logging.info("Loss is not defined")
    torch.save(args, save_path)

def load_meta_model(load_path, train=True):
    """
    Loading model
    args:
        load_path: string
    """
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    if 'args' in checkpoint:
        args = checkpoint['args']

    vocab = checkpoint['vocab']
    is_factorized = args.is_factorized
    r = args.r

    model = init_transformer_model(args, vocab, train=train, is_factorized=is_factorized, r=r)
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        print("CUDA")
        model = model.cuda()
    else:
        model = model.cpu()

    inner_opt = torch.optim.SGD(model.parameters(), lr=args.lr)
    outer_opt = torch.optim.Adam(model.parameters(), lr=args.meta_lr)
    inner_opt.load_state_dict(checkpoint['inner_opt'].state_dict())
    outer_opt.load_state_dict(checkpoint['outer_opt'].state_dict())

    return model, vocab, inner_opt, outer_opt, epoch, metrics, args

def load_joint_model(load_path, train=True):
    """
    Loading model
    args:
        load_path: string
    """
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    if 'args' in checkpoint:
        args = checkpoint['args']

    vocab = checkpoint['vocab']
    is_factorized = args.is_factorized
    r = args.r

    model = init_transformer_model(args, vocab, train=train, is_factorized=is_factorized, r=r)
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        print("CUDA")
        model = model.cuda()
    else:
        model = model.cpu()

    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    opt.load_state_dict(checkpoint['opt'].state_dict())

    return model, vocab, opt, epoch, metrics, args

def load_model(load_path, train=True):
    """
    Loading model
    args:
        load_path: string
    """
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

    epoch = checkpoint['epoch']
    metrics = checkpoint['metrics']
    if 'args' in checkpoint:
        args = checkpoint['args']

    vocab = checkpoint['vocab']
    is_factorized = args.is_factorized
    r = args.r

    # args.feat_extractor = "vgg_cnn"
    args.k_lr = 1
    args.min_lr = 1e-6

    model = init_transformer_model(args, vocab, train=train, is_factorized=is_factorized, r=r)
    model.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        print("CUDA")
        model = model.cuda()
    else:
        model = model.cpu()

    opt = init_optimizer(args, model)
    if opt is not None:
        opt.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        if args.loss == "ce":
            opt._step = checkpoint['optimizer_params']['_step']
            opt._rate = checkpoint['optimizer_params']['_rate']
            opt.warmup = checkpoint['optimizer_params']['warmup']
            opt.factor = checkpoint['optimizer_params']['factor']
            opt.model_size = checkpoint['optimizer_params']['model_size']
        elif args.loss == "ctc":
            opt.lr = checkpoint['optimizer_params']['lr']
            opt.lr_anneal = checkpoint['optimizer_params']['lr_anneal']
        else:
            print("Need to define loss type")
            logging.info("Need to define loss type")

    return model, vocab, opt, epoch, metrics, args

def load_discriminator(load_path, train=True):
    """
    Loading discriminator
    args:
        load_path: string
    """
    checkpoint = torch.load(load_path, map_location=torch.device('cpu'))

    epoch = checkpoint['epoch']
    if 'args' in checkpoint:
        args = checkpoint['args']

    discriminator = init_discriminator_model(args)
    discriminator.load_state_dict(checkpoint['model_state_dict'])
    if args.cuda:
        print("CUDA")
        discriminator = discriminator.cuda()
    else:
        discriminator = discriminator.cpu()

    opt = torch.optim.Adam(discriminator.parameters(), lr=args.lr)
    opt.load_state_dict(checkpoint['opt'].state_dict())

    return discriminator, opt

def init_optimizer(args, model, opt_type="noam"):
    dim_input = args.dim_input
    warmup = args.warmup
    lr = args.lr

    if opt_type == "noam":
        opt = NoamOpt(dim_input, args.k_lr, warmup, torch.optim.Adam(model.parameters(), betas=(0.9, 0.98), eps=1e-9), min_lr=args.min_lr)
    elif opt_type == "sgd":
        opt = AnnealingOpt(lr, args.lr_anneal, torch.optim.SGD(model.parameters(), lr=lr, momentum=args.momentum, nesterov=True))
    else:
        opt = None
        print("Optimizer is not defined")

    return opt

def init_transformer_model(args, vocab, train=True, is_factorized=False, r=100):
    """
    Initiate a new transformer object
    """
    if args.feat_extractor == 'emb_cnn':
        hidden_size = int(math.floor(
            (args.sample_rate * args.window_size) / 2) + 1)
        hidden_size = int(math.floor(hidden_size - 41) / 2 + 1)
        hidden_size = int(math.floor(hidden_size - 21) / 2 + 1)
        hidden_size *= 32
        args.dim_input = hidden_size
    elif args.feat_extractor == 'vgg_cnn':
        hidden_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1) # 161
        hidden_size = int(math.floor(int(math.floor(hidden_size)/2)/2)) * 128 # divide by 2 for maxpooling
        args.dim_input = hidden_size
        if args.feat == "logfbank":
            args.dim_input = 2560
    elif args.feat_extractor == 'large_cnn':
        hidden_size = int(math.floor((args.sample_rate * args.window_size) / 2) + 1) # 161
        hidden_size = int(math.floor(int(math.floor(hidden_size)/2)/2)) * 64 # divide by 2 for maxpooling
        args.dim_input = hidden_size
    else:
        print("the model is initialized without feature extractor")

    num_enc_layers = args.num_enc_layers
    num_dec_layers = args.num_dec_layers
    num_heads = args.num_heads
    dim_model = args.dim_model
    dim_key = args.dim_key
    dim_value = args.dim_value
    dim_input = args.dim_input
    dim_inner = args.dim_inner
    dim_emb = args.dim_emb
    src_max_len = args.src_max_len
    tgt_max_len = args.tgt_max_len
    dropout = args.dropout
    emb_trg_sharing = args.emb_trg_sharing
    feat_extractor = args.feat_extractor

    encoder = Encoder(num_enc_layers, num_heads=num_heads, dim_model=dim_model, dim_key=dim_key, dim_value=dim_value, dim_input=dim_input, dim_inner=dim_inner, src_max_length=src_max_len, dropout=dropout, is_factorized=is_factorized, r=r)
    decoder = Decoder(vocab, num_layers=num_dec_layers, num_heads=num_heads, dim_emb=dim_emb, dim_model=dim_model, dim_inner=dim_inner, dim_key=dim_key, dim_value=dim_value, trg_max_length=tgt_max_len, dropout=dropout, emb_trg_sharing=emb_trg_sharing, is_factorized=is_factorized, r=r)
    decoder = decoder if train else decoder
    model = Transformer(encoder, decoder, vocab, feat_extractor=feat_extractor, train=train)

    return model

def init_discriminator_model(args):
    dim_model = args.dim_model
    num_class = args.num_class
    discriminator = Discriminator(dim_model, num_class)

    return discriminator

def post_process(string, special_token_list):
    for i in range(len(special_token_list)):
        string = string.replace(special_token_list[i],"")
    string = string.replace("▁"," ")
    return string
