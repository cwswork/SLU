import gc
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn

from align import model_util
from autil.sparse_tensor import SpecialSpmm

class GAT_Layer(nn.Module):
    def __init__(self, kgs_data, myconfig):
        super(GAT_Layer, self).__init__()
        self.is_cuda = myconfig.is_cuda
        self.device = myconfig.device

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.PReLU() #nn.ReLU(inplace=True)
        self.special_spmm = SpecialSpmm()

        self.KG_E = kgs_data.KG_E
        #self.gcn_dropout = torch.nn.Dropout(p=myconfig.gcn_dropout, inplace=False)
        self.gat_Linear = nn.Linear(myconfig.input_dim, myconfig.out_dim)
        self.w_atten_r = nn.Parameter(torch.ones(size=(myconfig.out_dim*2, 1)), requires_grad=True)  # zeros->  ones
        model_util.init_params(self.w_atten_r)

        # Initialize parameters
        self.apply(lambda module: model_util.init_params(module)) #, n_layers=2

    # 2
    def forward(self, batch_ids, batch_adj_arr, ent_embed, feature_dropout):

        # 2 gat model
        batch_embed = self.gat_layer(ent_embed, batch_adj_arr, batch_ids, feature_dropout)
        return batch_embed

    def gat_layer(self, ent_embed, batch_adj_arr, batch_ids, feature_dropout):
        # feature_dropout
        batch_edge_index, batch_shape = self.drop_neigh(batch_adj_arr, batch_ids, feature_dropout)

        gat_in = self.gat_Linear(ent_embed)
        left_embed = gat_in[batch_ids, :]

        left_embed1 = left_embed[batch_edge_index[0, :], :]
        right_embed1 = gat_in[batch_edge_index[1, :], :]

        eer_embed = torch.cat((left_embed1, right_embed1), dim=1)  # 3d
        ee_atten = torch.exp(
            -self.leakyrelu(torch.matmul(eer_embed, self.w_atten_r).squeeze()))  # (D,2d)*(2d,1) => D

        dv = 'cuda' if self.is_cuda else 'cpu'
        e_rowsum = self.special_spmm(batch_edge_index, ee_atten, batch_shape,
                                     torch.ones(size=(self.KG_E, 1), device=dv))  # (B, E) (E,1) => (B,1)
        # e_out: attention*h = ee_atten * e_embed
        e_out = self.special_spmm(batch_edge_index, ee_atten, batch_shape, gat_in)  # (B,E)*(E,dim) = (B,dim)
        e_out = e_out * (1. / e_rowsum)

        return self.relu(e_out)  # (E,dim)

    def drop_neigh(self, sparse_mx, batch_ids, drop_prob):
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        if drop_prob != 0:
            all_list = list(range(len(sparse_mx.row)))
            get_index = sorted(random.sample(all_list, int(len(all_list) * (1 - drop_prob))), reverse=False)
            re_ids = np.array(get_index)

            row = sparse_mx.row[re_ids]
            col = sparse_mx.col[re_ids]
            #data = sparse_mx.data[re_ids]
        else:
            row, col, data = sparse_mx.row, sparse_mx.col, sparse_mx.data

        batch_ids = batch_ids.detach().cpu().numpy()
        node_num = np.arange(len(batch_ids))
        row = np.concatenate([row, node_num])
        col = np.concatenate([col, batch_ids])
        #data = np.concatenate([data, np.ones_like(diag)])

        adj_indexs = torch.from_numpy(
            np.vstack((row, col)).astype(np.int64))
        shape = torch.Size(sparse_mx.shape)
        if self.is_cuda:
            adj_indexs = adj_indexs.to(self.device)

        return adj_indexs, shape


