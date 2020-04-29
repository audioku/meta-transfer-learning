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

import pandas as pd

class Analyzer():
    """
    Trainer class
    """
    def __init__(self):
        logging.info("Analyzer is initialized")

    def analyze(self, train_loaders, valid_loaders, test_loaders, train_manifests, valid_manifests, test_manifests):
        """
        Training
        args:
            model: Model object
            train_loader: DataLoader object of the training set
            valid_loader_list: a list of Validation DataLoader objects
            opt: Optimizer object
            start_epoch: start epoch (> 0 if you resume the process)
            num_epochs: last epoch
        """
        data_dict = {
            'train': (train_loaders, train_manifests), 
            'valid': (valid_loaders, valid_manifests), 
            'test': (test_loaders, test_manifests)
        }
    
        for prefix, (data_loaders, data_manifests) in data_dict.items():
            for i, data_loader in enumerate(data_loaders):
                manifest = '{}_stats_{}.csv'.format(prefix, data_manifests[i].split('/')[-1].split('.')[0])
                print('MANIFEST {}'.format(manifest), flush=True)
                src_lens = []
                trg_lens = []
                for idx, (src, trg, src_percentages, src_lengths, trg_lengths) in enumerate(data_loader):

                    src_len = src_lengths.squeeze().tolist()
                    trg_len = trg_lengths.squeeze().tolist()
                    
                    # Handle single element
                    if not isinstance(src_len, list):
                        src_len = [src_len]
                    if not isinstance(trg_len, list):
                        trg_len = [trg_len]
                        
#                     print('SRC {}'.format(src_len), flush=True)
#                     print('TRG {}'.format(trg_len), flush=True)

                    src_lens += src_len
                    trg_lens += trg_len

       	        df = pd.DataFrame({'src':src_lens, 'trg':trg_lens})
                df.to_csv(manifest, index=False)

                print('{} MAX SRC'.format(manifest, df['src'].max()))
                print('{} MAX TRG'.format(manifest, df['trg'].max()))
                print('{} STATS'.format(manifest))
                print(df.describe([0.01,0.05,0.25,0.4,0.5,0.6,0.75,0.95,0.99]))
        return 0
