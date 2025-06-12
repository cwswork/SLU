import pickle
import numpy as np

def savepickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def loadpickle(path):
    with open(path, "rb+") as f:
        data = pickle.load(f)
    return data

def printNumpy(outfile, a):
    with open(outfile, 'w', encoding='utf-8') as fw:
        for i in range(a.shape[0]):
            ll = list(a[i])
            fw.write('{}\n'.format(ll.__str__()))

#########################################################
def save_dict2txt(outfile, out_dict, save_kv='kv'):
    with open(outfile, 'w', encoding='utf-8') as fw:
        if save_kv == 'vk':
            for (k, v) in out_dict.items():
                fw.write('{}\t{}\n'.format(v, k))
        else:
            for (k, v) in out_dict.items():
                fw.write('{}\t{}\n'.format(k, v))

def save_list2txt(outfile, triples_list):
    if len(triples_list)==0:
        return

    with open(outfile, 'w', encoding='utf-8') as fw:
        if len(triples_list[0]) == 2:
            for (a1, a2) in triples_list:
                fw.write('{}\t{}\n'.format(a1, a2))
        elif len(triples_list[0]) == 3:
            for (h, r, t) in triples_list:
                fw.write('{}\t{}\t{}\n'.format(h, r, t))
        else:
            for l in triples_list:
                fw.write('{}\n'.format(l))
                # fw.write(line.__str__() + '\n')


def save_triple2txt(outfile, triples_list):
    with open(outfile, 'w', encoding='utf-8') as fw:
        if len(triples_list[0]) == 3:
            for (h, r, t) in triples_list:
                fw.write('{}\t{}\t{}\n'.format(int(h), int(r), int(t)))
        elif len(triples_list[0]) == 2:
            for (a1, a2) in triples_list:
                fw.write('{}\t{}\n'.format(int(a1), int(a2)))
        else:
            print('wrong input')


#########################################################
def get_links2ids(links_file, kg1_ent2id_dict, kg2_ent2id_dict, isID=False):
    if isID:
        ent_links = read_links_ids(links_file)
    else:
        ent_links = load_list(links_file)

    links_ids = list()
    for (e1, e2) in ent_links:
        assert e1 in kg1_ent2id_dict
        assert e2 in kg2_ent2id_dict
        links_ids.append((kg1_ent2id_dict[e1], kg2_ent2id_dict[e2]))

    links_ids = np.array(links_ids)

    return links_ids

def read_links(file_path):
    '''
    :param file_path: such as 'datasets\D_W_15K_V1\721_5fold\1\train_links'
    :return: train_links
    '''
    print("read links:", file_path)
    links = list()
    refs = list()
    reft = list()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 2
        e1 = params[0].strip()
        e2 = params[1].strip()
        refs.append(e1)
        reft.append(e2)
        links.append((e1, e2))
    assert len(refs) == len(reft)
    return links

def read_links_ids(links_file):
    '''
     将连接转换会id列表:list:tuple(2)
    '''
    print("load_list:", links_file)
    links_ids_list = []
    with open(links_file, encoding='utf-8', mode='r') as f:
        for line in f:
            th = line[:-1].split('\t')
            links_ids_list.append((int(th[0]), int(th[1])))

    return links_ids_list


def load_list(file_path):
    '''
    :param file_path:
    :return:
    '''
    print("load_list:", file_path)
    new_list = []
    with open(file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            th = line[:-1].split('\t')
            new_list.append(th)

    return new_list


def load_dict(file_path, read_kv='kv', sep='\t'):
    '''
    :param file_path:
    :param read_kv: ='kv' or 'vk'
    :return:
    '''
    print("load dict:", file_path)
    value_trans_dict = {}
    with open(file_path, encoding='utf-8', mode='r') as f:
        if read_kv == 'kv':
            for line in f:
                th = line[:-1].split(sep)
                if len(th) == 2:
                    value_trans_dict[th[0]] = th[1]  # id:value
                else:
                    value_trans_dict[th[0]] = ' '.join(th[1:])
        else:  # vk
            for line in f:
                th = line[:-1].split(sep)
                value_trans_dict[th[1]] = th[0]  # value:id

    return value_trans_dict

def load_dict_vk(file_path, sep='\t'):
    print("load dict:", file_path)
    value_trans_dict = {}
    with open(file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            th = line[:-1].split(sep)
            value_trans_dict[th[1]] = int(th[0])  # value:id
    return value_trans_dict

def load_ids2list(file_path):
    '''
    :param file_path:
    :return:
    '''
    print("load ids to list file:", file_path)
    new_list = []
    with open(file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            th = line[:-1].split('\t')
            new_list.append((int(th[0]), th[1]))
    return new_list


def load_ids2dict(file, read_kv='kv', sep='\t'):
    # 实体，已编号  ent_ids_1、ent_ids_2
    print('loading ids_dict file ' + file)
    kg_ent2id_dict = dict()
    with open(file, encoding='utf-8') as f:
        if read_kv =='vk':
            for line in f:
                th = line[:-1].split(sep)
                kg_ent2id_dict[th[1]] = int(th[0])  # (name:eid)
        else:
            for line in f:
                th = line[:-1].split(sep)
                kg_ent2id_dict[int(th[0])] = th[1]  # (eid:name)

    return kg_ent2id_dict


def load_ids(file):
    # 实体，已编号  ent_ids_1、ent_ids_2
    print('loading a ent_ids...' + file)
    with open(file, encoding='utf-8') as f:
        kg_ent2id = []
        for line in f:
            th = line[:-1].split('\t')
            kg_ent2id.append(int(th[0])) # (eid)

    return kg_ent2id


def load_triples_id(file_path):
    triple_list = []
    with open(file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            th = line[:-1].split('\t')
            th = [int(i) for i in th]
            triple_list.append(th)

    return triple_list

def load_triples_id2(file_path):
    triple_list = []
    relSet = set()
    with open(file_path, encoding='utf-8', mode='r') as f:
        for line in f:
            th = line[:-1].split('\t')
            th = [int(i) for i in th]
            triple_list.append(th)
            relSet.add(th[1]) # relation

    return triple_list, relSet

# 读取文件
## 三元组 ###########################################
def read_relation_triples(file_path):
    '''
    read relation_triples
    :param file_path: such as 'datasets\D_W_15K_V1\rel_triples_1'
    :return: triples, entities, relations(h, r, t)
    '''
    print("\nread relation triples:", file_path)
    if file_path is None:
        return set(), set(), set()
    triples = set()
    entities, relations = set(), set()
    file = open(file_path, 'r', encoding='utf8')
    for line in file.readlines():
        params = line.strip('\n').split('\t')
        assert len(params) == 3
        h = params[0].strip()
        r = params[1].strip()
        t = params[2].strip()
        triples.add((h, r, t))
        entities.add(h)
        entities.add(t)
        relations.add(r)
    print("Number of entities:", len(entities))
    print("Number of relations:", len(relations))
    print("Number of relation triples:", len(triples))
    return triples, entities, relations

def relation_triple2ids(relation_triples, ent_ids_dict, rel_ids_dict):
    ''' 转换relation_triple为id对列表:(h, r, t) '''
    relation_triples_id = list()
    for h, r, t in relation_triples:
        assert h in ent_ids_dict
        assert r in rel_ids_dict
        assert t in ent_ids_dict
        relation_triples_id.append((ent_ids_dict[h], rel_ids_dict[r], ent_ids_dict[t]))
    assert len(relation_triples_id) == len(set(relation_triples))

    return relation_triples_id

def sort_elements(triples, elements_set):
    ''' 对实体进行频率排序'''
    dic = dict()
    for s, p, o in triples:
        if s in elements_set:
            dic[s] = dic.get(s, 0) + 1
        if p in elements_set:
            dic[p] = dic.get(p, 0) + 1
        if o in elements_set:
            dic[o] = dic.get(o, 0) + 1
    # firstly sort by values (i.e., frequencies), if equal, by keys (i.e, URIs)
    sorted_list = sorted(dic.items(), key=lambda x: (x[1], x[0]), reverse=True)
    ordered_elements = [x[0] for x in sorted_list]
    return ordered_elements, dic

def gen_mapping_id(kg1_triples, kg1_elements, kg2_triples, kg2_elements, ordered=True):
    '''
    【实体，关系】
    给两个kg的元素编号，kg1编偶数，kg2编奇数 '''
    kg1_ids, kg2_ids = dict(), dict()
    if ordered:
        kg1_ordered_elements, _ = sort_elements(kg1_triples, kg1_elements)
        kg2_ordered_elements, _ = sort_elements(kg2_triples, kg2_elements)
        n1 = len(kg1_ordered_elements)
        n2 = len(kg2_ordered_elements)
        n = max(n1, n2)
        for i in range(n):
            if i < n1 and i < n2:
                kg1_ids[kg1_ordered_elements[i]] = i * 2
                kg2_ids[kg2_ordered_elements[i]] = i * 2 + 1
            elif i < n1:
                kg1_ids[kg1_ordered_elements[i]] = n2 * 2 + (i - n2)
            elif i < n2:
                kg2_ids[kg2_ordered_elements[i]] = n1 * 2 + (i - n1)
    else:
        index = 0
        for ele in kg1_elements:
            if ele not in kg1_ids:
                kg1_ids[ele] = index
                index += 1
        for ele in kg2_elements:
            if ele not in kg2_ids:
                kg2_ids[ele] = index
                index += 1
    assert len(kg1_ids) == len(set(kg1_elements))
    assert len(kg2_ids) == len(set(kg2_elements))
    return kg1_ids, kg2_ids
