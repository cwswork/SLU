import copy
import math
import os
import pickle
import time
import random
from os.path import join

import numpy as np
import scipy
import torch

from autil import alignment

##################################
# 1 对齐路径
def get_alignRR(ent_embed, pseudo_link, ent_neigh_dict, kg_E, kg_R, pre_alignRel, neight_minsim, align_num_thre):
    # 2、计算训练集的实体对的评分数量——left_neigh_match
    max_neighbors_num = len(max(ent_neigh_dict.values(), key=lambda x: len(x)))  # 最大邻居
    print('Maximum number of neighbors:' + str(max_neighbors_num))  # max_neighbors_num = 235
    ent_pad_id, rel_pad_id = kg_E, kg_R  # 空白邻居的实体和关系ID
    left_neigh_match = neigh_match(ent_embed, pseudo_link, ent_neigh_dict,
                                                max_neighbors_num, ent_pad_id, rel_pad_id)

    # 3、根据训练集的实体对的评分数量，确定关系对匹配——RR_pair_dict
    RR_pair_dict, temp_RR_list, temp_notRR_list = rel_match(left_neigh_match, ent_pad_id, rel_pad_id, pre_alignRel, neight_minsim=neight_minsim, align_num_thre=align_num_thre) # 20
    return RR_pair_dict, temp_RR_list, temp_notRR_list

#2、计算训练集的实体对的评分数量——left_neigh_match
def neigh_match(ename_embed, pseudo_link, ent_neigh_dict_old, max_neighbors_num, ent_pad_id, rel_pad_id):
    """Similarity Score (between entity pairs)
        return: [B,neigh,5]: (tail_i, tail_j, rel_i, rel_j, ent_ij_sim)
    """
    ent_neigh_dict = copy.deepcopy(ent_neigh_dict_old)
    # 构建定长的邻居矩阵
    for e in ent_neigh_dict.keys():
        pad_list = [(rel_pad_id, ent_pad_id)] * (max_neighbors_num - len(ent_neigh_dict[e]))
        ent_neigh_dict[e] += pad_list

    dim = len(ename_embed[0])
    zero_embed = [0.0 for _ in range(dim)]  # <PAD> embedding
    ename_embed = np.vstack([ename_embed, zero_embed])
    ename_embed = torch.FloatTensor(ename_embed)

    print("pseudo_link (e1,e2) num is: {}".format(len(pseudo_link)))
    start_time = time.time()
    left_neigh_match, right_neigh_match = [], []
    batch_size = 1000
    for start_pos in range(0, len(pseudo_link), batch_size):  # len(ent_pairs)=750000
        end_pos = min(start_pos + batch_size, len(pseudo_link))
        batch_ent_pairs = pseudo_link[start_pos: end_pos]
        e1s = [e1 for e1, e2 in batch_ent_pairs]
        e2s = [e2 for e1, e2 in batch_ent_pairs]
        #ent_neigh_dict Key:(r, t)
        er1_neigh = np.array([ent_neigh_dict[e1] for e1 in e1s])  # size: [B(Batchsize),ne1(e1_neighbor_max_num)]
        er2_neigh = np.array([ent_neigh_dict[e2] for e2 in e2s])
        r1_neigh = er1_neigh[:, :, 0]  # r
        r2_neigh = er2_neigh[:, :, 0]
        e1_neigh = er1_neigh[:, :, 1]  # t
        e2_neigh = er2_neigh[:, :, 1]

        e1_neigh_tensor = torch.LongTensor(e1_neigh)  # [B,neigh]
        e2_neigh_tensor = torch.LongTensor(e2_neigh)
        e1_neigh_emb = ename_embed[e1_neigh_tensor]  # [B,neigh,embedding_dim]
        e2_neigh_emb = ename_embed[e2_neigh_tensor]

        #sim_mat_max = alignment.cosine_similarity3(e1_neigh_emb, e2_neigh_emb) # 越相似，值越大
        #max_scoce, max_index = sim_mat_max.topk(k=1, dim=-1, largest=True)  # 取前top_num最大
        # 越相似，值越小！！
        min_scoce, min_index = alignment.torch_sim_min_topk_s(e1_neigh_emb, e2_neigh_emb, top_num=1,
                                                              metric='cosine')
        min_scoce = min_scoce.squeeze(-1)  # [B,neigh,1] -> [B,neigh] #get max value.
        min_index = min_index.squeeze(-1)

        batch_match = np.zeros([e1_neigh.shape[0], e1_neigh.shape[1], 5])
        for e in range(e1_neigh.shape[0]): # [B,neigh]
            e1_array = e1_neigh[e] # [neigh,1] = >[neigh]
            e2_array = e2_neigh[e, min_index[e]]
            r1_array = r1_neigh[e]
            r2_array = r2_neigh[e, min_index[e]]
            scoce_array = min_scoce[e]
            if len(e2_array) != len(set(e2_array.tolist())): # 多对一
                for i in range(1, len(e2_array)):
                    if e1_array[i] == ent_pad_id: # 空邻居
                        break

                    if e2_array[i] in e2_array[0: i]: # 多对一, 这个对齐邻居之前出现过
                        index = np.where(e2_array[0: i] == e2_array[i])[0][0]
                        if scoce_array[index] < scoce_array[i]:
                            e2_array[i] = ent_pad_id # 前面(index)的小（越相似）,保留前面
                        else:
                            e2_array[index] = ent_pad_id # 后面(i)的小,保留后面
            #aa = np.vstack((e1_array, e2_array, r1_array, r2_array, scoce_array)).T
            batch_match[e] = np.vstack((e1_array, e2_array, r1_array, r2_array, scoce_array)).T

        if type(left_neigh_match) is np.ndarray:
            left_neigh_match = np.vstack((left_neigh_match, batch_match))
        else:
            left_neigh_match = batch_match  # [B,neigh,5]: (tail_i, tail_j, rel_i, rel_j, ent_ij_sim)

    print("all ent pair left_neigh_match shape:{}".format(left_neigh_match.shape) )
    print("using time {:.3f}".format(time.time() - start_time))
    return left_neigh_match #, right_neigh_match

#3、根据训练集的实体对的评分数量，确定关系对匹配——RR_pair_dict
def rel_match(left_neigh_match, ent_pad_id, rel_pad_id, pre_alignRel, neight_minsim, align_num_thre):
    rel_pair_score = {}
    num_max = 10**6
    for r1, r2 in pre_alignRel:
        rel_pair_score[(r1, r2)] = [1.0, num_max]

    ent_ij_sim5_num = 0
    pass_sim = 0
    print('==neight_minsim: ', neight_minsim) # 0.5
    print('==align_num_thre: ', align_num_thre)
    for neigh_ll in left_neigh_match.tolist():
        for (tail_i, tail_j, rel_i, rel_j, ent_ij_sim) in neigh_ll:
            tail_i, tail_j, rel_i, rel_j = int(tail_i), int(tail_j), int(rel_i), int(rel_j)
            if tail_i == ent_pad_id or tail_j == ent_pad_id: # or ent_ij_sim == 0.0
                continue
            if rel_i == rel_pad_id or rel_j == rel_pad_id:
                continue

            if ent_ij_sim > neight_minsim:
                ent_ij_sim5_num += 1
                continue
            else:
                pass_sim += 1

            if (rel_i, rel_j) not in rel_pair_score.keys():
                rel_pair_score[(rel_i, rel_j)] = [ent_ij_sim, 1]
            else:
                rel_pair_score[(rel_i, rel_j)][0] += ent_ij_sim  # 相似度叠加
                rel_pair_score[(rel_i, rel_j)][1] += 1 # 数量

    for rel_pair, (score, num) in rel_pair_score.items():
        if num >= num_max:
            rel_pair_score[rel_pair] = [score, num] # 平均相似度
        else:
            rel_pair_score[rel_pair] = [score/num, num] # 平均相似度


    print('ent_ij_sim5_num:' + str(ent_ij_sim5_num))
    print('pass_sim:' + str(pass_sim))

    print("all rel_pair_score len:" + str(len(rel_pair_score)))
    # 按“匹配数量 num ”递减排序
    sim_rank_order = sorted(rel_pair_score.items(), key=lambda kv: kv[1][1], reverse=True)
    RR_list, notRR_list = [], []   # list([r1_id, r2_id, sim_v, num])
    RR_pair_dict = dict()
    for (r1_id, r2_id), (sim_v, num) in sim_rank_order:
        if r1_id not in RR_pair_dict.keys() and r2_id not in RR_pair_dict.values():

            if num < align_num_thre: # 排除，匹配数量太少
                continue
            RR_pair_dict[r1_id] = r2_id
            RR_list.append([r1_id, r2_id, sim_v, num])
        else:
            notRR_list.append([r1_id, r2_id, sim_v, num])

    RR_list = sorted(RR_list, key=lambda kv: kv[0], reverse=False)  # 按照r1_id升序
    notRR_list = sorted(notRR_list, key=lambda kv: kv[0], reverse=False)
    return RR_pair_dict, RR_list, notRR_list


def get_pathadj(rel_triples, head_rt, tail_rh, KG_E, matchpath):  # 加入子链接
    ''' path(h, r, ..., r2, t) 拼接 head_rt'''

    #new_head_rt = {i:set() for i in range(KG_E)}  # h:(r,t)
    #new_tail_rt = {i:set() for i in range(KG_E)}  # h:(r,t)

    path_triples_count = 0
    path_adj = np.zeros((KG_E, KG_E))
    adj_max_len = 0
    for (h,r,t) in rel_triples: #(h, r, t)
        for r2, t2 in head_rt[t]:   # (h, r, t) + (t, r2, t2)
            newpath = (r, r2)
            if newpath in matchpath.keys():
                if h!=t2:
                    path_adj[h, t2] += 1
                    adj_max_len = max(path_adj[h, t2], adj_max_len)
                    #new_head_rt[h].add((match_path[newpath], t2))
                    path_triples_count+=1

        for r2, t2 in tail_rh[h]:  #  + (h2, r2, h) (h,r, t)
            newpath = (-r, -r2)
            if newpath in matchpath.keys():
                if h != t2:
                    path_adj[t, t2] += 1  # 双向
                    adj_max_len = max(path_adj[t, t2], adj_max_len)
                    #new_tail_rt[t].add((match_path[newpath], t2))
                    path_triples_count += 1

    print('add path_triples:{}, adj_max_len:{}'.format( path_triples_count, str(adj_max_len)))

    return path_adj
