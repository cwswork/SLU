import gc
import numpy as np
import torch
import torch.nn as nn

from align import model_util
from align.Lcat_Layer import Lcat_Layer
from align.gat_Layer import GAT_Layer
from align.gcngat_Layer import GCNGAT_Layer


class LCAT_model(nn.Module):
    def __init__(self, kgs_data, myconfig):
        super(LCAT_model, self).__init__()
        self.is_cuda = myconfig.is_cuda
        self.device = myconfig.device
        self.momentum = myconfig.momentum

        self.name_embed = kgs_data.name_embed
        if self.is_cuda:
            self.name_embed = self.name_embed.to(myconfig.device)
        myconfig.input_dim = len(self.name_embed[0])

        self.node_layer = model_util.FeedForwardNetwork(myconfig.input_dim, myconfig.out_dim) #nn.Linear
        self.rel_skip_w = nn.Parameter(torch.ones(1))

        #self.lcat_layer = Lcat_Layer(kgs_data, myconfig)
        if myconfig.mode == 'lcat':
            self.gnn_layer = Lcat_Layer(kgs_data, myconfig)
        elif myconfig.mode == 'gcngat':
            self.gnn_layer = GCNGAT_Layer(kgs_data, myconfig)
        elif myconfig.mode == 'gat':
            self.gnn_layer = GAT_Layer(kgs_data, myconfig)

        # Initialize parameters
        self.apply(lambda module: model_util.init_params(module))


    def update(self, network: nn.Module):
        for key_param, query_param in zip(self.parameters(), network.parameters()):
            key_param.data *= self.momentum
            key_param.data += (1 - self.momentum) * query_param.data
        self.eval()

    def forward(self, batch_ids, batch_adj, feature_dropout=0):
        if self.is_cuda:
            batch_ids = batch_ids.to(self.device)

        # 1 entity self
        node_out = self.node_layer(self.name_embed[batch_ids,:])  # (B,d)  node_layer
        # model gcn+gat
        lcat_out = self.gnn_layer(batch_ids, batch_adj, self.name_embed, feature_dropout)  # (B,d)

        ''' Add skip connection with learnable weight self.skip[t_id]'''

        alpha = torch.sigmoid(self.rel_skip_w) # Tanh
        output = lcat_out * alpha + node_out * (1 - alpha)

        return  output



