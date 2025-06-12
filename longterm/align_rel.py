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

from autil import fileUtil, alignment
from longterm import align_fun


def get_pseudo_labels(ent_embed1, ent_embed2, kg1_ent_ids, kg2_ent_ids, pseudo_minsim, all_IIL=None):
    # 右边不能划分
    score_mat_left, index_mat_left = alignment.torch_sim_min_topk_s(ent_embed1, ent_embed2, top_num=1,
                                                                    metric='cosine')
    score_mat_left, index_mat_left = score_mat_left.squeeze(-1), index_mat_left.squeeze(-1)

    score_mat_right, index_mat_right = alignment.torch_sim_min_topk_s(ent_embed2, ent_embed1, top_num=1,
                                                                      metric='cosine')
    score_mat_right, index_mat_right = score_mat_right.squeeze(-1), index_mat_right.squeeze(-1)

    IIL_list = []
    print('==pseudo_minsim: ', pseudo_minsim)
    for i, p in enumerate(index_mat_left):
        # 从左到右，从右到左都对齐
        if 1 - score_mat_left[i] < pseudo_minsim or 1 - score_mat_right[p] < pseudo_minsim:
            continue
        elif index_mat_right[p] == i:
            ee_pair = (kg1_ent_ids[i], kg2_ent_ids[p])
            # if new_links == [] or (ee_pair in new_links):  # 交集
            IIL_list.append(ee_pair)

    if len(IIL_list) > 0:
        if all_IIL !=None:
            union_IIL = set(IIL_list).intersection(set(all_IIL)) # 交集
            inIIL = len(union_IIL)
            print("==True IIL({})/ all IIL({}):{:.4f}% == finded IIL:{}. ".format(inIIL,
                              len(IIL_list), inIIL / len(IIL_list) * 100, len(all_IIL)))
        IIL_arr = np.array(IIL_list)
        return IIL_arr
    else:
        print("==no IIL")

        return None

##################################
def get_triple(rel_triples, KG_E):
    head_rt = { i:list() for i in range(KG_E)}  # h:(r,t)
    #tail_rh = { i:list() for i in range(KG_E)}  # h:(r,t)
    #rels = set()
    nei_adj = np.zeros((KG_E, KG_E))
    for (h, r, t) in rel_triples:
        if h != t:
            head_rt[h].append((r, t))
            head_rt[t].append((r, h))

            #tail_rh[t].append((-r, h))
            nei_adj[h, t] += 1
            nei_adj[t, h] += 1  # 双向
        #rels.add(r)
    #print('===rels:', len(rels))  # 关系的个数
    return head_rt, nei_adj#, tail_rh, len(rels)

def load_rel2dict(file, sep='\t'):
    # 实体，已编号  ent_ids_1、ent_ids_2
    print('loading ids_dict file ' + file)
    id2dict = dict()
    with open(file, encoding='utf-8') as f:
        for line in f:
            th = line[:-1].split(sep)
            id = int(th[0])
            name = th[1].split('/')[-1]
            id2dict[id] = name  # (eid:name)

    return id2dict

def get_rels(data_dir):
    # rel
    kg1_ent_ids = fileUtil.load_ids(join(data_dir, 'ent_ids_1'))
    kg2_ent_ids = fileUtil.load_ids(join(data_dir, 'ent_ids_2'))
    #ent_ids = kg1_ent_ids + kg2_ent_ids
    KG_E = len(kg1_ent_ids) + len(kg2_ent_ids)

    kg1_rel_dict = load_rel2dict(join(data_dir, 'rel_ids_1'))
    kg2_rel_dict = load_rel2dict(join(data_dir, 'rel_ids_2'))
    #ent_ids = kg1_ent_ids + kg2_ent_ids
    KG_R = len(kg1_rel_dict) + len(kg2_rel_dict)

    rel_triples1 = fileUtil.load_triples_id(join(data_dir, 'triples_1'))
    rel_triples2 = fileUtil.load_triples_id(join(data_dir, 'triples_2'))
    rel_triples = rel_triples1+rel_triples2
    ##########
    embed_dict1 = fileUtil.loadpickle(join(data_dir, "LaBSE_emb_1.pkl"))
    embed_dict2 = fileUtil.loadpickle(join(data_dir, "LaBSE_emb_2.pkl"))
    embed_dict1.update(embed_dict2)  # 将dict2合并进dict1
    name_embed = []
    for i in range(KG_E):
        name_embed.append(embed_dict1[i][0])
    name_embed = torch.FloatTensor(name_embed)

    print("Num of KG1 entitys:", len(kg1_ent_ids))
    print("Num of KG2 entitys:", len(kg2_ent_ids))
    print("Num of KG1 relations:", len(kg1_rel_dict))
    print("Num of KG2 relations:", len(kg2_rel_dict))
    print("Num of KGs rel triples:", len(rel_triples))

    # with open(data_dir + 'pre/kgs_num', 'w') as ff:
    #     ff.write('KG_E\t' + str(len(ent_dict)) + '\n')
    #     ff.write('KG_R\t' + str(len(rel_dict)) + '\n')

    return KG_E, KG_R, name_embed, kg1_ent_ids, kg2_ent_ids, kg1_rel_dict, kg2_rel_dict, rel_triples

##################################
def get_alignrel(data_dir, head_rt, pseudo_link, KG_E, KG_R, name_embed, pre_alignRel=[], neight_minsim=0.5, align_num_thre=20):
    # 2、计算训练集的实体对的评分数量，确定路径对匹配——path_pair_dict
    rel_pair_dict, alignRel_list, noalignRel_list = align_fun.get_alignRR(name_embed, pseudo_link,  head_rt, KG_E, KG_R, pre_alignRel, neight_minsim=neight_minsim, align_num_thre=align_num_thre)
    print("Number of rel_pair_dict:" + str(len(rel_pair_dict)))
    fileUtil.savepickle("{}/path2/alignRel_dict".format(data_dir), rel_pair_dict)
    fileUtil.save_list2txt(data_dir + 'path2/alignRel_list.txt', alignRel_list)
    fileUtil.save_list2txt(data_dir + 'path2/noalignRel_list.txt', noalignRel_list)

    alignRel_dict = dict()
    for r1, r2 in rel_pair_dict.items():
        alignRel_dict[r1] = 0
        alignRel_dict[r2] = 0

    # 根据匹配rel，重新筛选triples
    new_ent_adj = np.zeros((KG_E, KG_E))
    new_triple_num = 0
    for h, rt_list in head_rt.items():
        rt_list = set([ (r, t) for r, t in rt_list if r in alignRel_dict])
        #ent_neigh_dict[h] = rt_list
        for r, t in rt_list:
            new_ent_adj[h, t] += 1
            new_triple_num += 1
    print('Number of new_triple:' + str(new_triple_num))  # max_neighbors_num = 235

    #max_neighbors_num = len(max(ent_neigh_dict.values(), key=lambda x: len(x)))  # 最大邻居
    #print('Maximum number of neighbors After align rel:' + str(max_neighbors_num))  # max_neighbors_num = 235

    return new_ent_adj


def main_ent_adj(data_dir, neight_list, pseudo_minsim, neight_minsim, align_num_thre):
    # 1 获得基础数据
    KG_E, KG_R, name_embed, kg1_ent_ids, kg2_ent_ids, kg1_rel_dict, kg2_rel_dict, rel_triples = get_rels(data_dir)
    head_rt, ent_adj = get_triple(rel_triples, KG_E)
    ent_adj_csr = scipy.sparse.csr_matrix(ent_adj)
    fileUtil.savepickle("{}path2/ent_adj_csr".format(data_dir), ent_adj_csr)

    # !!! del 获得 pseudo_link
    kg1_ent_tensor = torch.LongTensor(kg1_ent_ids)
    kg2_ent_tensor = torch.LongTensor(kg2_ent_ids)
    all_IIL = fileUtil.read_links_ids(data_dir + 'ref_ent_ids')

    ## 获得伪对齐实体
    pseudo_link = get_pseudo_labels(name_embed[kg1_ent_tensor, :], name_embed[kg2_ent_tensor, :], kg1_ent_ids, kg2_ent_ids, pseudo_minsim =pseudo_minsim, all_IIL=all_IIL)
    fileUtil.savepickle("{}path2/pseudo_link".format(data_dir), pseudo_link)
    # 获得伪对齐关系
    pre_alignRel = []
    kg2_rel_name2id = { name:id  for id, name in kg2_rel_dict.items()}
    for kg1_id, name in kg1_rel_dict.items():
        if name in kg2_rel_name2id.keys():
            pre_alignRel.append((kg1_id, kg2_rel_name2id[name])) # rl_name = r2_name
    print('==pre_alignRel len:', len(pre_alignRel))

    # 获得匹配路径
    ent_adj_new = get_alignrel(data_dir, head_rt, pseudo_link, KG_E, KG_R, name_embed, pre_alignRel, neight_minsim=neight_minsim, align_num_thre=align_num_thre)
    sp = scipy.sparse.csr_matrix(ent_adj_new)
    fileUtil.savepickle("{}path2/ent_adj_new".format(data_dir), sp)

    # 统计原始邻居
    count_array = np.count_nonzero(ent_adj != 0, axis=1) # 找出所有不够邻居数的行（节点）
    print('1. ent_adj neight_len max', '--', max(count_array))
    add_entlist = np.where(count_array == 0 )[0]
    print('1. ent_adj neight_len=0', '--', len(add_entlist))

    #  统计 对齐rel后的邻居
    count_array = np.count_nonzero(ent_adj_new != 0, axis=1)
    print('2. ent_adj_new neight_len max', '--', max(count_array))
    add_entlist = np.where(count_array == 0 )[0] # # 找出所有不够邻居数的行（节点）
    print('2. ent_adj_new neight_len=0', '--', len(add_entlist))

    ##############
    for min_neight in neight_list:
        print('----------min_neight:{}----------'.format(min_neight))
        ent_adj_new2 = ent_adj_new.copy()
        add_entlist2 = np.where(count_array < min_neight)[0] # # 找出所有不够邻居数的行（节点）
        print('3. ent_adj_new2 neight_len<', min_neight, '--', len(add_entlist2))
        for ent_id in add_entlist2:
            ent_adj_new2[ent_id] = ent_adj_new2[ent_id] + ent_adj[ent_id]
        # 保存邻接矩阵
        sp = scipy.sparse.csr_matrix(ent_adj_new2)
        fileUtil.savepickle("{}path2/ent_adj_new{}".format(data_dir, min_neight), sp)

        #  统计 融合后的邻居
        count_array2 = np.count_nonzero(ent_adj_new2 != 0, axis=1)
        print('4. ent_adj_new2 neight_len max', '--', max(count_array2))
        add_entlist2 = np.where(count_array2 < min_neight )[0] # # 找出所有不够邻居数的行（节点）
        print('4. ent_adj_new2 neight_len<', min_neight, '--', len(add_entlist2))
        print('--------------')


def main_ent_adj_reset(data_dir, min_neight=2):
    # 1 获得基础数据
    ent_adj = fileUtil.loadpickle(data_dir + 'path2/ent_adj_csr')
    ent_adj = ent_adj.toarray()

    ent_adj_new = fileUtil.loadpickle(data_dir + 'path2/ent_adj_new')
    ent_adj_new = ent_adj_new.toarray()

    # 统计原始邻居
    print('----------min_neight:{}----------'.format(min_neight))
    count_array = np.count_nonzero(ent_adj != 0, axis=1) # 找出所有不够邻居数的行（节点）
    print('1. ent_adj neight_len max', '--', max(count_array))
    add_entlist = np.where(count_array == 0 )[0]
    print('1. ent_adj neight_len=0', '--', len(add_entlist))

    #  统计 对齐rel后的邻居
    count_array = np.count_nonzero(ent_adj_new != 0, axis=1)
    print('2. ent_adj_new neight_len max', '--', max(count_array))
    add_entlist = np.where(count_array == 0 )[0] # # 找出所有不够邻居数的行（节点）
    print('2. ent_adj_new neight_len=0', '--', len(add_entlist))

    ##############
    add_entlist = np.where(count_array < min_neight)[0] # # 找出所有不够邻居数的行（节点）
    print('3. ent_adj_new neight_len <', min_neight, '--', len(add_entlist))
    for ent_id in add_entlist:
        ent_adj_new[ent_id] = ent_adj_new[ent_id] + ent_adj[ent_id]
    # 保存邻接矩阵
    sp = scipy.sparse.csr_matrix(ent_adj_new)
    fileUtil.savepickle("{}path2/ent_adj_new{}".format(data_dir, min_neight), sp)

    #  统计 融合后的邻居
    count_array = np.count_nonzero(ent_adj_new != 0, axis=1)
    print('4. ent_adj_new2 neight_len max', '--', max(count_array))
    add_entlist = np.where(count_array < min_neight )[0] # # 找出所有不够邻居数的行（节点）
    print('4. ent_adj_new2 neight_len <', min_neight, '--', len(add_entlist))

def main_run(data_dir):
    if not os.path.exists(data_dir + 'path2/'):
        os.makedirs(data_dir + 'path2/')

    print("start==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    seed = 72
    print(data_dir)
    random.seed(seed)
    np.random.seed(seed)

    neight_minsim = 0.5
    align_num_thre = 10
    # zh_en ja_en  pseudo_minsim= 0.8
    # —— fr_en / EN_DE_15K_V1 / EN_FR_15K_V1  pseudo_minsim= 0.98
    if 'zh_en' in data_dir:
        pseudo_minsim = 0.8
    elif 'ja_en' in data_dir:  # —— ja_en
        pseudo_minsim = 0.8
        align_num_thre = 5
    elif 'fr_en' in data_dir:  # —— fr_en
        pseudo_minsim = 0.8
        align_num_thre = 5
    else:
        pseudo_minsim = 0.98

    main_ent_adj(data_dir, neight_list=[5, 3], pseudo_minsim=pseudo_minsim,
                 neight_minsim=neight_minsim, align_num_thre=align_num_thre)

    # main_ent_adj_reset(data_dir, 3)
    print("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
if __name__ == '__main__':

    # 数据集：包括属性triples【attr_triples_1、attr_triples_2】
    # 关系triples【rel_triples_1、rel_triples_2】
    # 参数配置
    data_dir = '../../datasets2024/'
    #d = 'DBP15K/zh_en/' # fr_en, ja_en, zh_en
    # #d = 'WN31/EN_FR_15K_V2/' # EN_DE_15K_V1、EN_FR_15K_V1
    # #d = 'DWY100K/dbp_wd/' # DWY100K/dbp_wd, DWY100K/dbp_yg
    #
    #main_run(data_dir + d)
    # main_ent_adj_reset(data_dir + d, 10)
    # main_ent_adj_reset(data_dir + d, 15)

    data_list = ['DWY100K/dbp_wd/', 'DWY100K/dbp_yg/']
    for d in data_list:
        main_run(data_dir + d)
        main_ent_adj_reset(data_dir + d,10)
        main_ent_adj_reset(data_dir + d,15)

    #data_list = ['DBP15K/ja_en/','DBP15K/fr_en/']  # 'DBP15K/zh_en/',
    # data_list = ['WN31/EN_DE_15K_V1/', 'WN31/EN_DE_15K_V2/','WN31/EN_FR_15K_V1/','WN31/EN_FR_15K_V2/',]
    # for d in data_list:
    #     #main_run(data_dir + d)
    #     main_ent_adj_reset(data_dir+d, 10)
    #     main_ent_adj_reset(data_dir+d, 15)
