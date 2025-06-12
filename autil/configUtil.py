from __future__ import division
from __future__ import print_function
import os
import random
import re
import time
import argparse

import json
import numpy as np
import torch

# Training settings # 创建 ArgumentParser()对象
def parse_args():
    """
    Generate a parameters parser.
    """

    # parse parameters
    parser = argparse.ArgumentParser() ## 创建 ArgumentParser()对象

    # main parameters  # 调用add_argument() 方法添加参数
    parser.add_argument('--name', type=str, default="")
    parser.add_argument('--dataset', type=str, default="")
    parser.add_argument('--division', type=str, default="")

    parser.add_argument('--device', type=str, default="0",
                        help='Device cuda id')
    parser.add_argument('--seed', type=int, default=72,
                        help='Random seed.')
    parser.add_argument('--top_k', type=list, default=[1, 3, 5, 10, 50, 100],
                        help=' ')
    parser.add_argument('--neg_k', type=int, default=50,
                        help='采样.')
    parser.add_argument('--metric', type=str, default="L1")

    # model parameters
    parser.add_argument('--e_dim', type=int, default=300,
                        help='entity embedding size')
    parser.add_argument('--pe_dim', type=int, default=15,
                        help='position embedding size')

    parser.add_argument('--hops', type=int, default=7,
                        help='Hop of neighbors to be calculated')

    # parser.add_argument('--hidden_dim', type=int, default=512,
    #                     help='Hidden layer size')
    parser.add_argument('--ffn_dim', type=int, default=64,
                        help='FFN layer size')
    parser.add_argument('--n_layers', type=int, default=1,
                        help='Number of Transformer layers')
    parser.add_argument('--n_heads', type=int, default=8,
                        help='Number of Transformer heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout')
    parser.add_argument('--attention_dropout', type=float, default=0.1,
                        help='Dropout in the attention layer')

    # training parameters
    parser.add_argument('--batch_size', type=int, default=1000,
                        help='Batch size')
    parser.add_argument('--epochs', type=int, default=2000,
                       help='Number of epochs to train.')
    #parser.add_argument('--tot_updates',  type=int, default=1000,
    #                    help='used for optimizer learning rate scheduling')
    parser.add_argument('--warmup_updates', type=int, default=400,
                        help='warmup steps')
    parser.add_argument('--peak_lr', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--end_lr', type=float, default=0.0001,
                        help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=5e-4, #0.00001,
                        help='weight decay')


    parser.add_argument('--start_valid', type=int, default=60,
                        help=' ')
    parser.add_argument('--eval_freq', type=int, default=10,
                        help=' ')
    parser.add_argument('--eval_save_freq', type=int, default=10,
                        help=' ')
    parser.add_argument('--patience', type=int, default=50,
                        help='Patience for early stopping')
    parser.add_argument('--patience_minloss', type=int, default=50,
                        help='Patience for early stopping')

    args = parser.parse_args(args=[])

    return args

class configClass():
    def __init__(self, args, run_file, default_path):

        self.time_str = time.strftime('%Y-%m-%d_%H_%M_%S', time.localtime(time.time()))
        self.datasetPath = default_path + args.dataset  # + 'pre2/'
        self.tt_path = self.datasetPath + '721_5fold/' + args.division
        self.outputPath = self.datasetPath + args.division + args.name + '/' + self.time_str + '/'

        # 将参数生成函数和成员
        self.set_myprint(run_file)  # 初始化,打印和日志记录

        all_args = ''
        attr = vars(args)  # 将解析值转换成字典对象，然后就可以使用了
        for k, v in attr.items():
            all_args +="{}:{}, ".format( str(k), str(v))
            setattr(self, k, v)

        # 运行设置
        random.seed(self.seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        torch.cuda.manual_seed_all(self.seed)
        if not os.path.exists(self.outputPath):
            os.makedirs(self.outputPath)

        self.is_cuda = (self.device != "-1") and torch.cuda.is_available()  # cuda是否可用
        self.device = torch.device("cuda:" + self.device if self.is_cuda else "cpu")
        self.myprint("start==" + self.time_str + ": " + run_file)
        self.myprint('outputPath:' + self.outputPath)
        self.myprint('cuda.is_available:{}, GPU:{}'.format(str(self.is_cuda), self.device))
        self.myprint(all_args)


    # 结果保存文件名
    def get_param(self):
        model_param = 'eps_' + str(self.epochs) + \
            '-nk_' + str(self.neg_k) + \
            '-me_' + str(self.metric) + \
            '-lr_' + str(self.peak_lr) + \
            '-ed_' + str(self.e_dim) + \
            '-hn_' + str(self.n_heads) + \
            '-ly_' + str(self.n_layers)
            # '-gr_' + str(self.gamma_rel) + \
            # '-lbe_' + str(self.l_beta) + \
            # '-pa_' + str(self.patience) + \
            # '-drop_' + str(self.dropout)
            # '-be2_' + str(self.beta2)
            # '-be1_' + str(self.beta1) + \

        return model_param

    def set_myprint(self, runfile, issave=True):
        if issave:
            print_Class = Myprint(self.outputPath, 'train_log' + self.time_str + '.txt')
            if not os.path.exists(self.outputPath):
                print('outputPath not exists' + self.outputPath)
                os.makedirs(self.outputPath)
            self.myprint = print_Class.print
        else:
            self.myprint = print


        #self.myprint('model arguments:' + self.get_param())
        #self.myprint('modelChannel:' + str(self.modelChannel))

    #####################################
class Myprint:
    def __init__(self, filePath, filename):
        if not os.path.exists(filePath):
            print('outputPath not exists' + filePath)
            os.makedirs(filePath)

        self.outfile = filePath + filename

    def print(self, print_str):
        print(print_str)
        '''保存log文件'''
        with open(self.outfile, 'a', encoding='utf-8') as fw:
            fw.write('{}\n'.format(print_str))

#############################
class ARGs:
    ''' 加载配置问卷 args/** .json '''
    def __init__(self, file_path):
        args_dict = loadmyJson(file_path)  # 可以去除注释信息后加载json文件
        for k, v in args_dict.items():
            setattr(self, k, v)

################################################
# Load JSON File
def loadmyJson(JsonPath):
    try:
        srcJson = open(JsonPath, 'r', encoding= 'utf-8')
    except:
        print('cannot open ' + JsonPath)
        quit()

    dstJsonStr = ''
    for line in srcJson.readlines():
        if not re.match(r'\s*//', line) and not re.match(r'\s*\n', line):
            dstJsonStr += cleanNote(line)

    # print dstJsonStr
    dstJson = {}
    try:
        dstJson = json.loads(dstJsonStr)
    except:
        print(JsonPath + ' is not a valid json file')

    return dstJson


# 删除“//”标志后的注释，用于处理从文件中读出的字符串
def cleanNote(line_str):
    qtCnt = cmtPos = 0

    rearLine = line_str
    # rearline: 前一个“//”之后的字符串，
    # 双引号里的“//”不是注释标志，所以遇到这种情况，仍需继续查找后续的“//”
    while rearLine.find('//') >= 0: # 查找“//”
        slashPos = rearLine.find('//')
        cmtPos += slashPos
        headLine = rearLine[:slashPos]
        while headLine.find('"') >= 0:  # 查找“//”前的双引号
            qtPos = headLine.find('"')
            if not isEscapeOpr(headLine[:qtPos]):  # 如果双引号没有被转义
                qtCnt += 1 # 双引号的数量加1
            headLine = headLine[qtPos+1:]
            # print qtCnt
        if qtCnt % 2 == 0: # 如果双引号的数量为偶数，则说明“//”是注释标志
            # print self.instr[:cmtPos]
            return line_str[:cmtPos]
        rearLine = rearLine[slashPos+2:]
        # print rearLine
        cmtPos += 2

    return line_str


# 判断是否为转义字符
def isEscapeOpr(instr):
    if len(instr) <= 0:
        return False
    cnt = 0
    while instr[-1] == '\\':
        cnt += 1
        instr = instr[:-1]
    if cnt % 2 == 1:
        return True
    else:
        return False
