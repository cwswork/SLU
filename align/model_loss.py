import random
import numpy as np
import scipy.sparse as sp
import torch
import torch.nn as nn
import torch.nn.functional as F

from autil import alignment

class Nag_Loss():
    def __init__(self, myconfig):
        super(Nag_Loss, self).__init__()
        #self.device = myconfig.device
        self.loss_t = myconfig.loss_t
        self.CrossEntropyLoss = nn.CrossEntropyLoss()

    '''点乘'''
    def embed_sim__(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1) # (默认是2范数)
        z2 = F.normalize(z2)
        # z1 = F.normalize(z1, p=1.0) # (默认是2范数)
        sim_mat = torch.mm(z1, z2.t())
        return sim_mat

    def nce_loss(self, z1: torch.Tensor, z2: torch.Tensor): # semi_loss
        f = lambda x: torch.exp(x / self.loss_t)
        refl_sim = alignment.sim_cosine(z1, z1)  # (E1,d)*(d,E1)=>(E1,E1)
        between_sim = alignment.sim_cosine(z1, z2) #(E1,d)*(d,E2)=>(E1,E2)

        refl_sim = f(refl_sim)
        between_sim = f(between_sim)

        loss = -torch.log(between_sim.diag() / (between_sim.nansum(1) + refl_sim.nansum(1) - refl_sim.diag()) )
        if torch.isnan(torch.exp(loss)).sum() > 0:# 统计包含的空值总数
            print('-------loss_nan:{}------'.format(str(torch.isnan(torch.exp(loss)).sum())))

        return loss.nanmean() # 在存在 NaN 的情况下，torch.mean() 会将 NaN 传播到输出


    def simCLR_loss__(self,  pos_1, pos_2):
        out = torch.cat([pos_1, pos_2], dim=0)
        row_size = len(pos_1)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.loss_t)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * row_size, device=sim_matrix.device)).bool()
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * row_size, -1)

        # compute loss
        pos_sim = torch.exp(torch.sum(pos_1 * pos_2, dim=-1) / self.loss_t)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = - torch.log(pos_sim / sim_matrix.sum(dim=-1))
        return loss.sum()
