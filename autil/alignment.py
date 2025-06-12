import gc
import multiprocessing
from time import perf_counter

import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import functional
from autil import fileUtil, divide


##accuracy#
def get_hits(Left_vec, Right_vec, tt_links, top_k, metric, bata, LeftRight='Left'):
    #min_index, sim_mat = torch_sim_min_topk(Left_vec, Right_vec, metric=metric, bata=bata, top_num=top_k[-1])
    min_index, sim_mat = torch_sim_min_topk(Left_vec, Right_vec, metric=metric, bata=bata, top_num=top_k[-1])

    # left
    mr = 0
    mrr = 0
    tt_num = len(tt_links)
    all_hits = [0] * len(top_k)
    Hits_list = list()
    noHits1_num = 0
    # From left
    for row_i in range(min_index.shape[0]):
        e2_ranks_index = min_index[row_i, :].tolist() # row_i 行
        if row_i in e2_ranks_index:  # 测试
            rank_index = e2_ranks_index.index(row_i)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    all_hits[j] += 1
        else:
            row_mat_index = sim_mat[row_i].argsort().tolist()
            rank_index = row_mat_index.index(row_i)
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        # [Left, e1, e2, 与e1的对齐的hits1_e2，对齐排序], [Right, e2, e1, 与e2的对齐的hits1_e1，对齐排序]
        e1_gold, e2_gold = tt_links[row_i]
        hits1_e2 = tt_links[e2_ranks_index[0], 1]  # rank[0]取排名最高的
        Hits_list.append((LeftRight, e1_gold, e2_gold, hits1_e2, rank_index))
        if rank_index != 0:
            noHits1_num += 1

    assert len(Hits_list) == tt_num
    all_hits = [round(hh / tt_num * 100, 4) for hh in all_hits]
    mr /= tt_num  # 所有比对中等级的平均值
    mrr /= tt_num  # 所有倒数排名的均值

    # From left
    all_hits2 = '[{}]'.format('\t'.join([str(i) for i in all_hits]))
    #top_k2 = '\t'.join([str(i) for i in top_k])
    result_str1 = "Hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, noHits1:{}".format(
        top_k, all_hits2, mr, mrr, noHits1_num)

    return [all_hits, result_str1, Hits_list]

def get_hits_simple(Left_vec, Right_vec, top_k, metric, bata):
    min_index, sim_mat = torch_sim_min_topk(Left_vec, Right_vec, metric=metric, bata=bata, top_num=top_k[-1])

    # left
    mr = 0
    mrr = 0
    tt_num = len(Left_vec)
    all_hits = [0] * len(top_k)
    # From left
    for row_i in range(min_index.shape[0]):
        e2_ranks_index = min_index[row_i, :].tolist() # row_i 行
        if row_i in e2_ranks_index:  # 测试
            rank_index = e2_ranks_index.index(row_i)
            for j in range(len(top_k)):
                if rank_index < top_k[j]:
                    all_hits[j] += 1
        else:
            row_mat_index = sim_mat[row_i].argsort().tolist() #直接对top_k个相似值进行排序
            rank_index = row_mat_index.index(row_i)
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)

    all_hits = [round(hh / tt_num * 100, 4) for hh in all_hits]
    mr /= tt_num  # 所有比对中等级的平均值
    mrr /= tt_num  # 所有倒数排名的均值

    # From left
    all_hits2 = '[{}]'.format('\t'.join([str(i) for i in all_hits]))
    result_str1 = "Hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}".format(
        top_k, all_hits2, mr, mrr)
    return [all_hits, result_str1, []]


##similarity, topk########
def torch_sim_min_topk(embed1, embed2, metric='manhattan', bata=0.5, top_num=1, isdetach=True):
    ''' 返回越相似，值越小！ '''

    sim_mat = sim_min_batch(embed1, embed2, metric, bata)

    if len(embed2) < top_num:
        top_num = len(embed2)
    min_scoce_topk, min_index_topk = sim_mat.topk(k=top_num, dim=-1, largest=False)  # 取前top_num最小

    if isdetach:
        min_index_topk = min_index_topk.int()
        if sim_mat.is_cuda:
            sim_mat = sim_mat.detach().cpu().numpy()
            min_index_topk = min_index_topk.detach().cpu().numpy()
        else:
            sim_mat = sim_mat.detach().numpy()
            min_index_topk = min_index_topk.detach().numpy()

    return min_index_topk, sim_mat

def torch_sim_min_topk_s(embed1, embed2, metric='manhattan', bata=0.5, top_num=1, isdetach=True):
    ''' 返回越相似，值越小！ '''
    sim_mat = sim_min_batch(embed1, embed2, metric, bata)

    if len(embed2) < top_num:
        top_num = len(embed2)
    min_scoce_topk, min_index_topk = sim_mat.topk(k=top_num, dim=-1, largest=False)  # 取前top_num最小

    if isdetach:
        min_index_topk = min_index_topk.int()
        if sim_mat.is_cuda:
            min_scoce_topk = min_scoce_topk.detach().cpu().numpy()
            min_index_topk = min_index_topk.detach().cpu().numpy()
        else:
            min_scoce_topk = min_scoce_topk.detach().numpy()
            min_index_topk = min_index_topk.detach().numpy()

    return min_scoce_topk, min_index_topk

# 允许三维向量
def sim_min_batch(embed1, embed2, metric='manhattan', bata=0.5):
    if metric == 'cosine': #  cosine 余弦相似度, 越相似，距离（返回值）越大
        sim_mat = 1 - sim_cosine(embed1, embed2)  # 返回尺寸[batch, net1, net1]
    elif metric == 'CDistance': #
        sim_mat = sim_CDistance(embed1, embed2, bata)
    # elif metric == 'CDistance_cosine':  #
    #     sim_mat = sim_CDistance_cosine(embed1, embed2)
    elif metric == 'L1' or metric == 'manhattan':  # L1 Manhattan 曼哈顿距离
        sim_mat = torch.cdist(embed1, embed2, p=1.0)  # 越相似，距离（返回值）越小，所以1-
    else: ## metric == 'L2' or metric == 'euclidean':  # L2 euclidean 欧几里得距离
        sim_mat = torch.cdist(embed1, embed2, p=2.0)

    return sim_mat

def sim_CDistance(embed1, embed2, bata=0.5):
    sim_mat_s = torch.mm(embed1, embed2.t()) #(E，E)
    #sim_mat_t = torch.mm(embed2, embed1.t())
    sim_mat_t =  sim_mat_s.T

    es_max_scoce, es_max_index = torch.max(sim_mat_s, dim=-1)  # 取每行相似度最大
    et_max_scoce, et_max_index = torch.max(sim_mat_t, dim=-1)  # 取每列相似度最大
    es2 = es_max_scoce.unsqueeze(1)
    div = bata * (es2 + et_max_scoce) # (E,E)
    sim = div - sim_mat_s
    return sim # sim 越大，值越大

def sim_CDistance_cosine(embed1, embed2, bata=0.5):
    embed1 = F.normalize(embed1, dim=-1)  # F.normalize只能处理两维的数据，L2归一化
    embed2 = F.normalize(embed2, dim=-1)
    if len(embed2.shape) == 3: # 矩阵是三维
        sim_mat_s = torch.bmm(embed1, torch.transpose(embed2, 1, 2))
        sim_mat_t = torch.transpose(sim_mat_s, 1, 2)

        es_max_scoce, es_max_index = torch.max(sim_mat_s, dim=-1)  # 取每行相似度最大
        et_max_scoce, et_max_index = torch.max(sim_mat_t, dim=-1)  # 取每列相似度最大
        es_max = es_max_scoce.unsqueeze(1)
        et_max = et_max_scoce.unsqueeze(-1)

        div = bata * (es_max + et_max) # (E,E)
        sim = div - sim_mat_s
    else:
        sim_mat_s = torch.mm(embed1, embed2.t())
        #sim_mat_t = torch.mm(embed2, embed1.t())
        sim_mat_t =  sim_mat_s.T

        es_max_scoce, es_max_index = torch.max(sim_mat_s, dim=-1)  # 取每行相似度最大
        et_max_scoce, et_max_index = torch.max(sim_mat_t, dim=-1)  # 取每列相似度最大
        es_max = es_max_scoce.unsqueeze(1)
        div = bata * (es_max + et_max_scoce) # (E,E)
        sim = div - sim_mat_s

    return sim # sim 越大，值越大

def sim_cosine(embed1, embed2):
    embed1 = F.normalize(embed1, dim=-1)  # F.normalize只能处理两维的数据，L2归一化
    embed2 = F.normalize(embed2, dim=-1)
    if len(embed2.shape) == 3: # 矩阵是三维
        sim = torch.bmm(embed1, torch.transpose(embed2, 1, 2))
    else:
        sim = torch.mm(embed1, embed2.t())
    return sim

def sim_CDistance_neight_(embed1, embed2, tt_adj):
    sim_mat_s = torch.mm(embed1, embed2.t()) #(E1，E2)
    sim_mat_t = torch.mm(embed2, embed1.t()) #(E2，E1)
    # 只保留邻居节点
    adj_mask = torch.where(tt_adj == 0, 0, 1)
    if embed1.is_cuda:
        adj_mask = adj_mask.cuda()

    s_mean = sim_mean(sim_mat_s, adj_mask)
    t_mean = sim_mean(sim_mat_t, adj_mask.t())

    # non_zero_indices = torch.nonzero(tt_adj)
    # sim_mat_s2 = sim_mat_s[non_zero_indices]
    # s_mean = torch.mean(sim_mat_s2, dim=-1, keepdim=True)
    #
    # sim_mat_t2 = sim_mat_t[non_zero_indices.t()]
    # t_mean = torch.mean(sim_mat_t2, dim=-1)

    s_mean = s_mean.unsqueeze(1)
    div = 0.5 * (s_mean + t_mean) # (E,E)
    sim = (sim_mat_s - div)
    return sim

def sim_mean(sim_mat_s, adj_mask):
    sim_mat_s = sim_mat_s * adj_mask  # mul

    es_sum = torch.sum(sim_mat_s, dim=-1)  #
    es_num = torch.count_nonzero(sim_mat_s, dim=-1)
    es_num = 1. / torch.where(es_sum!=0, es_num, 1)[0]
    re = es_sum * es_num
    return  re

def cal_sim(sim_mat, k):
    # 越相似，sim_mat（点积） 值越大（[-1,1]），cosine 值越大
    sorted_mat = -np.partition(-sim_mat, k + 1, axis=1) #如果kth是正值，排序结果保证了前kth个元素是最小的。在kth后面的这堆数值，都是这个向量里面比较大的群体们
    nearest_k = sorted_mat[:, 0:k]
    sim_values = np.mean(nearest_k, axis=1)
    return sim_values

def Neighbor(sim_mat1):  # (E，E)

    tasks = divide.task_divide(np.array(range(sim_mat1.shape[0])), 10)  # E n等分
    pool = multiprocessing.Pool(processes=len(tasks))
    reses = list()
    for task in tasks:
        reses.append(pool.apply_async(cal_sim, (sim_mat1[task, :], 1)))
    pool.close()
    pool.join()
    sim_values = None
    for res in reses:
        val = res.get()
        if sim_values is None:
            sim_values = val
        else:
            sim_values = np.append(sim_values, val)
    assert sim_values.shape[0] == sim_mat1.shape[0]
    return sim_values



def cosine_similarity3(a, b):
    '''
    a shape: [num_item_1, embedding_dim]
    b shape: [num_item_2, embedding_dim]
    return sim_matrix: [num_item_1, num_item_2]
    '''

    a = F.normalize(a, dim=-1)  # F.normalize只能处理两维的数据，L2归一化
    b = F.normalize(b, dim=-1)
    if len(b.shape) == 3: # 矩阵是三维
        sim = torch.bmm(a, torch.transpose(b, 1, 2))
    else:
        sim = torch.mm(a, b.t())

    return sim


