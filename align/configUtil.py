from __future__ import division
from __future__ import print_function
import os
import random
import re
import time
import argparse

import json
from os.path import join

import numpy as np
import torch

#def parse_args():
def parse_args():
    """
    Generate a parameters parser.
    """
    # parse parameters
    parser = argparse.ArgumentParser() ## 创建 ArgumentParser()对象

    # main parameters  # 调用add_argument() 方法添加参数
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--Data_Dir', type=str, default="../../datasets2024/")
    parser.add_argument('--Out_Dir', type=str, default="../../output2024/")
    parser.add_argument('--ent_adj_file', type=str, default='')
    #parser.add_argument('--path_adj_file', type=str, default='')
    #parser.add_argument('--isPath', type=bool, default=False)

    ##parser.add_argument('--test_path', type=str, default=test_path)
    parser.add_argument('--name_embed', type=str, default='LaBSE_emb')
    parser.add_argument('--longterm_emb', type=str, default='') #longterm_LaBSE.embed
    parser.add_argument('--mode', type=str, default='lcat')

    # 新增 3.12
    parser.add_argument('--n_layers', type=int, default=1)
    parser.add_argument('--n_heads', type=int, default=1,
                        help='Number of Transformer heads')
    parser.add_argument('--hidden_dim', type=int, default=768)
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')
    parser.add_argument('--ffn_dropout', type=float, default=0.1,
                        help='ffn_dropout')

    #parser.add_argument('--division', type=str, default="")
    #parser.add_argument('--neg_k', type=int, default=50, help='采样.')
    parser.add_argument('--device', type=str, default="0",  help='Device cuda id')
    parser.add_argument('--seed', type=int, default=72, help='Random seed.')
    parser.add_argument('--top_k', type=list, default=[1, 3, 5, 10, 50, 100], help=' ')
    parser.add_argument('--metric', type=str, default="CDistance") # cosine，L1 CDistance
    parser.add_argument('--test_GPU', type=bool, default=True)

    #########
    # model parameters
    parser.add_argument('--input_dim', type=int, default=768, help='entity embedding size') # 768+15 = 783
    parser.add_argument('--out_dim', type=int, default=768,  help='Output layer size')
    parser.add_argument('--loss_t', type=float, default=0.08, help='')
    parser.add_argument('--momentum', type=float, default=0.9999, help='')
    parser.add_argument('--feature_dropout', type=float, default=0.2, help='Dropout feature in trainning') # 0.2 0.1
    parser.add_argument('--gcn_dropout', type=float, default=0.2, help='Dropout in the gcn layer') # 0.2 0.1
    parser.add_argument('--bata', type=float, default=0.5, help='')

    #parser.add_argument('--hops', type=int, default=1,  help='Hop of neighbors to be calculated') # 7
    #parser.add_argument('--pe_dim', type=int, default=0,  help='position embedding size') #15
    #parser.add_argument('--loss_choice', type=str, default='infoNCE', help='') #semi, infoNCE, infoNCE_KoLeo
    #parser.add_argument('--bi_loss', type=bool, default=True, help='') #BiInfoNCE

    #parser.add_argument('--n_heads', type=int, default=1, help='Number of Transformer heads')
    #parser.add_argument('--n_layers', type=int, default=1,  help='Number of Transformer layers')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size') # 5000 , 1024
    parser.add_argument('--train_epochs', type=int, default=800, help='Number of epochs to train.') # 2000
    parser.add_argument('--weight_decay', type=float, default=1e-5, help='weight decay')
    parser.add_argument('--peak_lr', type=float, default=5e-5, help='learning rate') # 0.001 5e-5
    # DBP15K:1e-4 //  5e-5
    # parser.add_argument('--warmup_updates', type=int, default=100,  help='warmup steps')
    # parser.add_argument('--tot_updates',  type=int, default=400,  help='used for optimizer learning rate scheduling')
    #parser.add_argument('--end_lr', type=float, default=1e-05, help='learning rate')

    parser.add_argument('--early_stop', type=bool, default=True, help=' ')
    parser.add_argument('--eval_freq', type=int, default=10, help=' ')
    parser.add_argument('--start_valid', type=int, default=10, help=' ')
    parser.add_argument('--patience', type=int, default=20, help='Patience for early stopping')
    #parser.add_argument('--eval_save_freq', type=int, default=10, help=' ')
    # parser.add_argument('--patience_minloss', type=int, default=20, help='Patience for early stopping')

    args = parser.parse_args(args=[])

    return args

class configClass():
    def __init__(self, args, run_dir):
        self.time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
        args.Data_Dir = join(args.Data_Dir,  args.dataset)
        args.Out_Dir = join(args.Out_Dir, args.dataset, self.time_str +'_' +args.name) + '/'
        os.makedirs(args.Out_Dir, exist_ok=True)

        self.all_args = ''
        for k, v in vars(args).items():# vars(args) 将解析值转换成字典对象
            self.all_args += "{}:{}, ".format( str(k), str(v))
            setattr(self, k, v)

        # Running set
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        self.is_cuda = (self.device != "cpu") and torch.cuda.is_available()  # cuda是否可用
        self.device = torch.device("cuda:" + self.device if self.is_cuda else "cpu")

        self.set_myprint()
        self.myprint("start==" + self.time_str + ": " + run_dir)
        self.myprint('Out_Dir:' + self.Out_Dir)
        self.myprint('cuda.is_available:{}, GPU:{}'.format(str(self.is_cuda), self.device))
        self.myprint("########all_args##########")
        self.myprint(self.all_args)
        self.myprint("##########################")

    def set_myprint(self, issave=True):
        if issave:
            print_Class = Myprint(self.Out_Dir, 'train_log' + self.time_str + '.txt')
            if not os.path.exists(self.Out_Dir):
                print('Out_Dir not exists' + self.Out_Dir)
                os.makedirs(self.Out_Dir)
            self.myprint = print_Class.print
        else:
            self.myprint = print

    #####################################
class Myprint:
    def __init__(self, filePath, filename):
        self.outfile = filePath + filename

    def print(self, print_str):
        print(print_str)
        '''log file'''
        with open(self.outfile, 'a', encoding='utf-8') as fw:
            fw.write('{}\n'.format(print_str))
