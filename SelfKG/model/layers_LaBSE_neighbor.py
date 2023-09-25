# coding: UTF-8
import os
from posixpath import join
import torch
import torch.nn as nn
from torch.nn import *
import torch.nn.functional as F
import numpy as np
import torch.optim as optim
from settings import *
import torch.utils.data as Data
from loader.DBP15KRawNeighbors import DBP15KRawNeighbors
from script.preprocess.deal_raw_dataset import MyRawdataset
import random
import faiss
import pandas as pd
import argparse
import logging
from datetime import datetime
# using labse
# from transformers import *
import torch


# Labse embedding dim
MAX_LEN = 88

def parse_options(parser):
    parser.add_argument('--device', type=str, default='cuda:0')
    parser.add_argument('--time', type=str, default=datetime.now().strftime("%Y%m%d%H%M%S"))
    parser.add_argument('--language', type=str, default='zh_en')
    parser.add_argument('--model_language', type=str, default='zh_en')
    parser.add_argument('--model', type=str, default='LaBSE')

    parser.add_argument('--epoch', type=int, default=300)
    parser.add_argument('--batch_size', type=int, default=64)
    parser.add_argument('--queue_length', type=int, default=64)

    parser.add_argument('--center_norm', type=bool, default=False)
    parser.add_argument('--neighbor_norm', type=bool, default=True)
    parser.add_argument('--emb_norm', type=bool, default=True)
    parser.add_argument('--combine', type=bool, default=True)

    parser.add_argument('--gat_num', type=int, default=1)

    parser.add_argument('--t', type=float, default=0.08)
    parser.add_argument('--momentum', type=float, default=0.9999)
    parser.add_argument('--lr', type=float, default=1e-6)
    parser.add_argument('--dropout', type=float, default=0.3)

    return parser.parse_args()

def adjust_learning_rate(optimizer, epoch, lr):
    if (epoch+1) % 10 == 0:
        lr *= 0.5
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

class MyEmbedder(nn.Module):
    def __init__(self, args, vocab_size, padding=ord(' ')):
        super(MyEmbedder, self).__init__()

        self.args = args

        self.device = torch.device(self.args.device)

    def forward(self, batch):
        batch = batch.to(self.device)
        
        return batch

class Trainer(object):
    def __init__(self, training=True, seed=37):
        # # Set the random seed manually for reproducibility.
        self.seed = seed
        fix_seed(seed)

        # set
        parser = argparse.ArgumentParser()
        self.args = parse_options(parser)

        self.device = torch.device(self.args.device)

        loader1 = DBP15KRawNeighbors(self.args.language, "1")
        myset1 = MyRawdataset(loader1.id_neighbors_dict, loader1.id_adj_tensor_dict)
        del loader1
        
        self.loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=True,
        )

        self.eval_loader1 = Data.DataLoader(
            dataset=myset1,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        del myset1

        loader2 = DBP15KRawNeighbors(self.args.language, "2")
        myset2 = MyRawdataset(loader2.id_neighbors_dict, loader2.id_adj_tensor_dict)
        del loader2

        self.loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=True,
        )

        self.eval_loader2 = Data.DataLoader(
            dataset=myset2,  # torch TensorDataset format
            batch_size=self.args.batch_size,  # all test data
            shuffle=True,
            drop_last=False,
        )

        del myset2

        self.model = None
        self.iteration = 0

        # get the linked entity ids
        def link_loader(mode, valid=False):
            link = {}
            if valid == False:
                f = 'test.ref'
            else:
                f = 'valid.ref'
            link_data = pd.read_csv(join(join(DATA_DIR, 'DBP15K', mode), f), sep='\t', header=None)
            link_data.columns = ['entity1', 'entity2']
            entity1_id = link_data['entity1'].values.tolist()
            entity2_id = link_data['entity2'].values.tolist()
            for i, _ in enumerate(entity1_id):
                link[entity1_id[i]] = entity2_id[i]
                link[entity2_id[i]] = entity1_id[i]
            return link

        self.link = link_loader(self.args.language)
        self.val_link = link_loader(self.args.language, True)

        def print_data_from_loaders(self):
            print("Printing data from loader1:")
            for batch_id, (language_id, token_data, id_data) in enumerate(self.loader1):
                print(f"Batch {batch_id}, Language ID: {language_id}")
                # Print token_data and id_data here as needed

            print("\nPrinting data from loader2:")
            for batch_id, (language_id, token_data, id_data) in enumerate(self.loader2):
                print(f"Batch {batch_id}, Language ID: {language_id}")
                # Print token_data and id_data here as needed

        def output(self):
            self.print_data_from_loaders()

