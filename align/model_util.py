import math
import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
from torch_geometric.nn.inits import glorot

def normalize_adj(mx):
    """Row-column-normalize sparse matrix"""
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1/2).flatten()
    r_inv[np.isinf(r_inv)] = 0.
    r_mat_inv = sp.diags(r_inv)
    mx = r_mat_inv.dot(mx).dot(r_mat_inv)
    return mx

def init_params(module, n_layers=1):
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=0.0, std=0.02 / math.sqrt(n_layers))
        if module.bias is not None:
            module.bias.data.zero_()

    if isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=0.02)

    if isinstance(module, nn.Parameter):
        glorot(module)
        #torch.nn.init.normal_(module, mean=0.0, std=0.02)

class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, out_size): #hidden_size, ffn_size
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, hidden_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x


class CostTimeMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self, name, fmt=":6.5f"):  #
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.train_cost = 0
        self.loss_sum = 0
        self.loss_count = 0

        self.acc_cost = 0
        self.acc_sum = [0.0]*6
        self.acc_count = 0
        # self.avg = 0

    def update_loss(self, epochs_i, loss, cost_t):
        self.train_cost += cost_t  # 总时间减去loss time
        self.loss_sum += loss
        self.loss_count += 1
        fmtstr = "Epoch-{:04d}: {}_loss:{"+self.fmt+"}, cost time:{:.4f}s"
        return fmtstr.format(epochs_i, self.name, loss, cost_t)

    def update_acc(self, epochs_i, hits_all, hits_str, cost_t):
        self.acc_cost += cost_t
        self.acc_sum = [ a+b for a, b in zip(self.acc_sum, hits_all) ]
        self.acc_count += 1

        fmtstr = "Epoch-{:04d}: {}-{}, cost time:{:.4f}s"
        return fmtstr.format(epochs_i, self.name, hits_str, cost_t)

    def get_avg_loss(self):
        fmtstr = "{}_AVG== Loss:{"+self.fmt+"}, cost time:{:.4f}s"
        return fmtstr.format(self.name, self.loss_sum/self.loss_count
                             , self.train_cost/self.loss_count)

    def get_avg_acc(self):
        fmtstr = "{}_AVG== Hits:{}, cost time:{:.4f}s"
        if self.acc_count!=0:
            acc_avg = [ i/self.acc_count for i in self.acc_sum if i!=0]
            acc_cost = self.acc_cost/self.acc_count
            return fmtstr.format(self.name, acc_avg, acc_cost)
        else:
            return fmtstr.format(self.name, 0, 0)

    # def __str__(self):
    #     fmtstr = "{name}==Loss_time:{" + self.fmt + "}, Acc_time:{" + self.fmt + "})"
    #     return fmtstr.format(self.loss_sum/self.loss_count, self.acc_sum/self.acc_count)


