import gc
import time

import numpy as np
import torch
import torch.nn.functional as F

from torch.nn import functional
from autil import fileUtil, alignment


##accuracy#
def get_hits1_iter(Left_vec, Right_vec, tt_links, top_k, metric, LeftRight='Left'):

    # From Left, 只取排名第一的top_num=1
    left_all_mat = alignment.sim_min_batch(Left_vec, Right_vec, metric=metric)
    # From right
    right_all_mat = left_all_mat.t()

    # 一对一
    _, index_mat_left_tensor = left_all_mat.topk(k=top_k[-1], dim=-1, largest=False)  # 取前top_num最小
    _, index_mat_right = right_all_mat.topk(k=top_k[-1], dim=-1, largest=False)  # 取前top_num最小
    if index_mat_left_tensor.is_cuda:
        index_mat_left = index_mat_left_tensor.detach().cpu().numpy()
        index_mat_right = index_mat_right.detach().cpu().numpy()
    else:
        index_mat_left = index_mat_left_tensor.detach().numpy()
        index_mat_right = index_mat_right.detach().numpy()

    ILL_hits1_dict = dict()
    left_ent, right_ent = tt_links[:, 0], tt_links[:, 1]
    for row_i in range(len(tt_links)):
        row_rank_indexs = index_mat_left[row_i]
        # 从左到右，从右到左都对齐
        if index_mat_right[row_rank_indexs[0]][0] == row_i: # 一对一
            # ent pair ID
            #ij_pair = (row_i, row_rank_indexs[0])
            ee_pair = (int(left_ent[row_i]), int(right_ent[row_rank_indexs[0]]))
            ILL_hits1_dict[ee_pair] = right_ent[row_rank_indexs].tolist()

    ILL_hits1_list = []
    noILL_dict = dict()
    for row_i, ee_pair in enumerate(tt_links):
        ee_pair = tuple(ee_pair)
        if ee_pair in ILL_hits1_dict.keys():
            ILL_hits1_list.append(ee_pair)
        else:
            noILL_dict[row_i] = ee_pair
    inILL = len(ILL_hits1_list) - len(noILL_dict)
    print("==True ILL({})/ all IIL({}):{:.4f}%. no Hits@1:{}. ".format(inILL,
                     len(ILL_hits1_list), inILL / len(ILL_hits1_list) * 100, len(noILL_dict)))

    ###############
    noILL_ids = list(noILL_dict.keys())
    noILL_ids = torch.LongTensor(np.array(noILL_ids))
    no_sim_mat = index_mat_left_tensor[noILL_ids, :]
    _, min_index_mat = no_sim_mat.topk(k=top_k[-1], dim=-1, largest=False)  # 取前top_num最小
    if min_index_mat.is_cuda:
        min_index_mat = min_index_mat.detach().cpu().numpy()
    else:
        min_index_mat = min_index_mat.detach().numpy()
    for rowid, ee_pair in enumerate(noILL_dict.values()):
        right_indexs = min_index_mat[rowid]
        ILL_hits1_dict[ee_pair] = noILL_ids[right_indexs].tolist()

    # left
    mr = 0
    mrr = 0
    tt_num = len(tt_links)
    all_hits = [0] * len(top_k)
    Hits_list = list()
    noHits1_num = 0
    for row_i, ee_pair in enumerate(tt_links):
        (e1_gold, e2_gold) = tuple(ee_pair)
        e2_ranks_list = ILL_hits1_dict[(e1_gold, e2_gold)]
        if e2_gold in e2_ranks_list:  # 测试
            rank_index = e2_ranks_list.index(e2_gold)
        else:
            row_mat_index = left_all_mat[row_i].argsort().tolist()
            rank_index = row_mat_index.index(row_i)

        for j in range(len(top_k)):
            if rank_index < top_k[j]:
                all_hits[j] += 1
        mr += (rank_index + 1)
        mrr += 1 / (rank_index + 1)
        # [Left, e1, e2, 与e1的对齐的hits1_e2，对齐排序], [Right, e2, e1, 与e2的对齐的hits1_e1，对齐排序]
        Hits_list.append((LeftRight, e1_gold, e2_gold, e2_ranks_list[0], rank_index))
        if rank_index != 0:
            noHits1_num += 1

    assert len(Hits_list) == tt_num
    all_hits = [round(hh / tt_num * 100, 4) for hh in all_hits]
    mr /= tt_num  # 所有比对中等级的平均值
    mrr /= tt_num  # 所有倒数排名的均值
    # From left
    result_str1 = "hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, noHits1:{}".format(
        top_k, all_hits, mr, mrr, noHits1_num)
    return [all_hits[0], result_str1, Hits_list]  # Left_re



#############################
def get_ent_score(rel_triples, kg_E, kg_R, metric):
    # self.ent_neigh_dict， self.rel_triples
    Rht_dict, Pr_dict, Prinv_dict = dict(), dict(), dict()
    # 关系的比例
    for rid in range(kg_R):
        Rht_dict[rid], Pr_dict[rid], Prinv_dict[rid] = set(), set(), set()
    for h, r, t in rel_triples:
        Rht_dict[r].add((h, t))
        Pr_dict[r].add(h)
        Prinv_dict[r].add(t)
    for rid in range(kg_R):
        if len(Rht_dict[rid]) > 0:
            Pr_dict[rid] = len(Pr_dict[rid]) / len(Rht_dict[rid])
            Prinv_dict[rid] = len(Prinv_dict[rid]) / len(Rht_dict[rid])
        else:
            Pr_dict[rid] = Prinv_dict[rid] = 0

    # 头尾节点的比例
    Ph_dict, Pt_dict = dict(), dict()
    for e in range(kg_E):
        Ph_dict[e] = Pt_dict[e] = 0
    for h, r, t in rel_triples:
        Ph_dict[h] += 1
        Pt_dict[t] += 1
    for e in range(kg_E):
        enum = Ph_dict[e] + Pt_dict[e]
        if enum > 0:
            Ph_dict[e] = Ph_dict[h] * 1.0 / enum
            Pt_dict[e] = Pt_dict[h] * 1.0 / enum
        else:
            Ph_dict[e] = Pt_dict[e] = 0
    # 头尾节点比例 * 关系比例
    head_score_dict, tail_score_dict = dict(), dict()
    for e in range(kg_E):
        head_score_dict[e] = tail_score_dict[e] = 0
    for h, r, t in rel_triples:
        head_score_dict[h] += Ph_dict[h] * Pr_dict[r]
        tail_score_dict[t] += Pt_dict[t] * Prinv_dict[r]

    # 最后得分Score_ent_dict，头节点的比例/尾节点的比例
    # ent_score_list = list()
    # for e in range(kg_E):
    #     if tail_score_dict[e] != 0:
    #         ent_score_list.append(-9e15)
    #     else:
    #         ent_score_list.append(head_score_dict[h] * 1.0 / tail_score_dict[t])

    head_score_list = sorted(head_score_dict.items(), key=lambda x:x[0], reverse=False)
    head_score = torch.FloatTensor(np.array(head_score_list))
    head_mat = alignment.sim_min_batch(head_score, head_score, metric=metric)

    tail_score_list = sorted(tail_score_dict.items(), key=lambda x: x[0], reverse=False)
    tail_score = torch.FloatTensor(np.array(tail_score_list))
    tail_mat = alignment.sim_min_batch(tail_score, tail_score, metric=metric)

    if min_index.is_cuda:
        min_index = min_index.detach().cpu().numpy()
    else:
        min_index = min_index.detach().numpy()
    return head_mat, tail_mat


def get_hits_new(ent_score_tensor, Left_vec, Right_vec, tt_links, top_k, metric, LeftRight='Left'):
    #t_begin = time.time()
    vec_mat = alignment.sim_min_batch(Left_vec, Right_vec, metric=metric)

    tt_links_tensor = torch.LongTensor(tt_links)
    Left_es = ent_score_tensor[tt_links_tensor[:, 0]]
    Right_es = ent_score_tensor[tt_links_tensor[:, 1]]
    es_mat = alignment.sim_min_batch(Left_es, Right_es, metric=metric)
    all_mat = vec_mat + 0.3 * es_mat
    return get_hits_bymat(all_mat, tt_links, top_k, LeftRight=LeftRight)


def get_hits_bymat(all_mat, tt_links, top_k, LeftRight='Left'):

    min_scoce, min_index = all_mat.topk(k=top_k[-1], dim=-1, largest=False)  # 取前top_num最小
    min_index = min_index.int()
    if min_index.is_cuda:
        min_index = min_index.detach().cpu().numpy()
    else:
        min_index = min_index.detach().numpy()

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
            row_mat_index = all_mat[row_i].argsort().tolist()
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
    result_str1 = "hits@{} = {}%, mr = {:.3f}, mrr = {:.6f}, noHits1:{}".format(
        top_k, all_hits, mr, mrr, noHits1_num)
    #print("get_hits_each: {:.6f}s".format(time.time() - t_begin))
    return [all_hits[0], result_str1, Hits_list] #Left_re

