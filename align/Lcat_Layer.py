import gc
import math
import random
import time
import numpy as np
import torch
import torch.nn as nn

from align import model_util
from autil.sparse_tensor import SpecialSpmm

class Lcat_Layer(nn.Module):
    def __init__(self, kgs_data, myconfig):
        super(Lcat_Layer, self).__init__()
        self.is_cuda = myconfig.is_cuda
        self.device = myconfig.device

        self.leakyrelu = nn.LeakyReLU(0.2)
        self.relu = nn.PReLU() #nn.ReLU(inplace=True)
        self.special_spmm = SpecialSpmm()  # sparse matrix multiplication

        self.KG_E = kgs_data.KG_E
        self.edge_index = kgs_data.edge_index
        self.edge_weight = kgs_data.edge_weight
        #self.edge_weight = torch.ones(size=(self.edge_index.shape[1], 1)).squeeze()

        ######## GCN
        self.gcn_dropout = torch.nn.Dropout(p=myconfig.gcn_dropout, inplace=False)
        #self.isgcn_Linear = myconfig.isgcn_Linear
        #if self.isgcn_Linear:
        #self.gcn_Linear = nn.Linear(myconfig.input_dim, myconfig.input_dim)

        self.encoder =EncoderLayer(myconfig.hidden_dim, myconfig.hidden_dim, myconfig.attention_dropout, myconfig.ffn_dropout,
                     myconfig.n_heads)

        self.lmbda12_ = nn.Parameter(torch.tensor([math.log10(6), 2.2 - math.log10(6)]), requires_grad=True)

        ######## GAT
        self.gat_Linear = nn.Linear(myconfig.out_dim, myconfig.out_dim) # Linear
        self.w_atten_r = nn.Parameter(torch.ones(size=(myconfig.out_dim*2, 1)), requires_grad=True)  # zeros->  ones
        model_util.init_params(self.w_atten_r)

        if self.is_cuda:
            self.edge_index = self.edge_index.to(myconfig.device)
            self.edge_weight = self.edge_weight.to(myconfig.device)
        # Initialize parameters
        self.apply(lambda module: model_util.init_params(module)) #, n_layers=2

    # 2
    def forward(self, batch_ids, batch_adj_arr, ent_embed, feature_dropout):
        # 1 GCN model
        gcn_out = self.gcn_layer(ent_embed)

        # 2 gat model
        gcn_out = self.gcn_dropout(gcn_out)
        batch_embed = self.gat_layer(gcn_out, batch_adj_arr, batch_ids, feature_dropout)
        return batch_embed

    def gcn_layer(self, ent_embed):
        # e_inlayer = self.dropout(e_inlayer)
        # if self.isgcn_Linear:
        #     gcn_in = self.gcn_Linear(ent_embed)
        # else:
        #     gcn_in = ent_embed
        #gcn_in = self.gcn_Linear(ent_embed)
        gcn_in = self.encoder(ent_embed)

        neigh_sum = self.special_spmm(self.edge_index, self.edge_weight, (self.KG_E, self.KG_E), gcn_in)

        dv = 'cuda' if self.is_cuda else 'cpu'
        rowsum = self.special_spmm(self.edge_index, self.edge_weight, (self.KG_E, self.KG_E),
                                   torch.ones(size=(self.KG_E, 1), device=dv))  # (E,E)*(E,1)=>(E,1)

        lmbda1_ = self.lmbda12_[0]
        lmbda2_ = self.lmbda12_[1]

        gcn_out = ((gcn_in + lmbda2_ * neigh_sum) / (1. + lmbda2_ * rowsum)) * lmbda1_

        return gcn_out

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


#################################
class EncoderLayer(nn.Module):
    def __init__(self, hidden_size, ffn_size, attention_dropout, ffn_dropout, n_heads):
        super(EncoderLayer, self).__init__()

        self.self_attention_norm = nn.LayerNorm(hidden_size) #??
        self.self_attention = MultiHeadAttention(
            hidden_size, attention_dropout, n_heads)
        self.self_attention_dropout = nn.Dropout(attention_dropout)

        self.ffn_norm = nn.LayerNorm(hidden_size)
        self.ffn = FeedForwardNetwork(hidden_size, ffn_size)
        self.ffn_dropout = nn.Dropout(ffn_dropout)

class MultiHeadAttention(nn.Module):
    def __init__(self, hidden_size, attention_dropout, n_heads):
        super(MultiHeadAttention, self).__init__()

        self.n_heads = n_heads
        self.att_size = hidden_size // n_heads
        self.scale = self.att_size ** -0.5
        attout_size = n_heads * self.att_size

        self.linear_q = nn.Linear(hidden_size, attout_size)
        self.linear_k = nn.Linear(hidden_size, attout_size)
        self.linear_v = nn.Linear(hidden_size, attout_size)
        self.att_dropout = nn.Dropout(attention_dropout)

        self.output_layer = nn.Linear(attout_size, hidden_size)

    def forward(self, q, k, v, attn_bias=None):
        orig_q_size = q.size()
        batch_size = q.size(0)

        # head_i = Attention(Q(W^Q)_i, K(W^K)_i, V(W^V)_i)
        q = self.linear_q(q).view(batch_size, -1, self.n_heads, self.att_size) # <B, hop, heads, d>
        k = self.linear_k(k).view(batch_size, -1, self.n_heads, self.att_size)
        v = self.linear_v(v).view(batch_size, -1, self.n_heads, self.att_size)

        q = q.transpose(1, 2)  # <B, heads, hop, d>
        v = v.transpose(1, 2)  # <B, heads, hop, d>
        k = k.transpose(1, 2).transpose(2, 3)  # <B, heads, d, hop>

        # Scaled Dot-Product Attention.
        # Attention(Q, K, V) = softmax((QK^T)/sqrt(d_k))V
        q = q * self.scale
        x = torch.matmul(q, k)  # <B, heads, hop, hop>
        if attn_bias is not None:
            x = x + attn_bias

        x = torch.softmax(x, dim=3)
        #x = self.att_dropout(x)
        x = x.matmul(v)  # <B, heads, hop, d>

        x = x.transpose(1, 2).contiguous()  #<B, hop, heads, d>
        x = x.view(batch_size, -1, self.n_heads * self.att_size)  #<B, hop, heads* d>

        x = self.output_layer(x)

        assert x.size() == orig_q_size
        return x


class FeedForwardNetwork(nn.Module):
    def __init__(self, hidden_size, ffn_size):
        super(FeedForwardNetwork, self).__init__()

        self.layer1 = nn.Linear(hidden_size, ffn_size)
        self.gelu = nn.GELU()
        self.layer2 = nn.Linear(ffn_size, hidden_size)

    def forward(self, x):
        x = self.layer1(x)
        x = self.gelu(x)
        x = self.layer2(x)
        return x
