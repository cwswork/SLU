import math
import os
import random
from os.path import join

import gensim
import scipy.sparse as sp
import numpy as np
import torch
import pickle

from align import model_util
from autil import fileUtil

class load_KGs_data(object):
    def __init__(self, myconfig):
        # Load Datasets
        ent_ids_1 = fileUtil.load_ids(join(myconfig.Data_Dir, 'ent_ids_1'))
        ent_ids_2 = fileUtil.load_ids(join(myconfig.Data_Dir, 'ent_ids_2'))
        KG1_E = len(ent_ids_1)
        KG2_E = len(ent_ids_2)
        self.KG_E = KG1_E + KG2_E
        self.ent_ids_list = ent_ids_1 + ent_ids_2
        # rel_triples1 = fileUtil.load_triples_id(join(myconfig.Data_Dir, 'triples_1'))
        # rel_triples2 = fileUtil.load_triples_id(join(myconfig.Data_Dir, 'triples_2'))
        # rel_triples = rel_triples1 + rel_triples2
        myconfig.myprint("Num of KG1 entitys:" + str(KG1_E))
        myconfig.myprint("Num of KG2 entitys:" + str(KG2_E))
        # myconfig.myprint("Num of KG1 rel_triples:" + str(len(rel_triples1)))
        # myconfig.myprint("Num of KG2 rel_triples:" + str(len(rel_triples2)))

        ### ent name embedding
        embed_dict1 = fileUtil.loadpickle(myconfig.Data_Dir + myconfig.name_embed + "_1.pkl")
        embed_dict2 = fileUtil.loadpickle(myconfig.Data_Dir + myconfig.name_embed + "_2.pkl")
        embed_dict1.update(embed_dict2)
        ent_embed = []
        for i in range(self.KG_E):
            ent_embed.append(embed_dict1[i][0])
        self.name_embed = torch.FloatTensor(ent_embed)
        ### longterm embedding
        if myconfig.longterm_emb!='':
            # if 'Word2vec' in myconfig.longterm_emb:
            #     Longterm_dict = gensim.models.KeyedVectors.load_word2vec_format(myconfig.Data_Dir + myconfig.longterm_emb)
            #     Longterm_embed = list()
            #     zero_embed = np.random.uniform(low=-1, high=1, size=len(Longterm_dict['0']))
            #     for e in range(self.KG_E):
            #         if str(e) not in Longterm_dict:
            #             print(e)
            #             Longterm_embed.append(zero_embed)
            #         else:
            #             #tt = Longterm_dict[str(e)]
            #             Longterm_embed.append(Longterm_dict[str(e)])
            #             Longterm_embed = torch.FloatTensor(Longterm_embed)
            Longterm_embed = fileUtil.loadpickle(myconfig.Data_Dir + myconfig.longterm_emb)

            self.name_embed = torch.cat((self.name_embed, Longterm_embed), dim=-1)


        adj_array_csr = fileUtil.loadpickle(join(myconfig.Data_Dir, myconfig.ent_adj_file))
        self.edge_index, self.edge_weight = get_adj_array(adj_array_csr)  # 用于GCN
        #self.edge_index, self.edge_weight = get_adj_array(adj_array_csr, isnormalized=False)  # 用于GCN, 不归一化??

        ### Val and Test data
        # val_path = myconfig.Data_Dir+ 'valid.ref' #+ 'valid.ref'  valid_links_id
        # test_path = myconfig.Data_Dir + 'test.ref' #+ 'test.ref'  test_links_id
        val_path = myconfig.Data_Dir + 'link/valid_links_id'
        test_path = myconfig.Data_Dir + 'link/test_links_id'
        val_ids_list, val_ar = get_links_loader(val_path)
        test_ids_list, self.test_link = get_links_loader(test_path)
        self.train_set = get_batchData(adj_array_csr, self.ent_ids_list, myconfig.batch_size)
        self.val_set = get_batchData(adj_array_csr, val_ids_list, myconfig.batch_size)
        self.test_set = get_batchData(adj_array_csr, test_ids_list, myconfig.batch_size)

    def initial_name_embed(self):
        tt_link = torch.LongTensor(self.test_link)
        return self.name_embed[tt_link[:, 0],:], self.name_embed[tt_link[:, 1],:]


def get_batchData(adj_array_csr, ids_list, batch_size=0):
    if batch_size !=0:
        batch_ids_list = task_divide(ids_list, batch_size)
    else:
        batch_ids_list = [ids_list]

    batch_list = []
    for batch_ids in batch_ids_list:
        #Bsize = len(batch_ids)
        mat_csr = adj_array_csr[batch_ids, :]  # sp.coo_matrix
        #Batch_adj_arr = get_batch_adj_array(mat_csr, batch_ids)  # 用于GAT
        batch_list.append((torch.LongTensor(batch_ids), mat_csr))

    return batch_list


def get_adj_array(sparse_mx, isnormalized=True):
    if isnormalized:
        sparse_mx = model_util.normalize_adj(sparse_mx).tocoo().astype(np.float32)
    else:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)

    row, col, weight = sparse_mx.row, sparse_mx.col, sparse_mx.data

    #adj_arr = np.vstack((row, col, weight))
    edge_index = torch.LongTensor(np.vstack((row, col)))
    edge_weight = torch.FloatTensor(weight)
    return edge_index, edge_weight


def get_links_loader(link_file):
    link_ls = fileUtil.read_links_ids(link_file)
    link_ar = np.array(link_ls)
    link_ids_list = link_ar[:,0].tolist() + link_ar[:,1].tolist()

    # link_left = link_ar[:,0].tolist()
    # link_right = link_ar[:,1].tolist()
    # mat_csr = adj_array_csr[link_left, :]
    # mat_csr = mat_csr[:, link_right]

    return link_ids_list, link_ar # , link_ls

def task_divide(idx, batch_size):
    ''' 划分成N个任务 '''
    total = len(idx)
    if total <= batch_size:
        return [idx]
    else:
        tasks = []
        beg =0
        while(beg<total):
            end = beg + batch_size
            if end<total:
                tasks.append(idx[beg:end])
            else:
                tasks.append(idx[beg:])

            beg = end

        return tasks

