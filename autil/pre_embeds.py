import os
import string
import time
import numpy as np
import pandas as pd

from gensim.models.word2vec import Word2Vec

import autil.fileUtil


##  实体名 嵌入 ##############################
def get_word_embed(entity_name_list):

    namesFrame = pd.DataFrame(entity_name_list)  # DataFrame，和excel表很像，names：(30000，2)
    names_len = len(namesFrame)
    print('namesFrame:', len(namesFrame))
    namesFrame.iloc[:, 1] = namesFrame.iloc[:, 1].str.replace(r'[pip {}]+'.format(string.punctuation), ' ').str.split(' ')  # 第3列改为list

    entity_name_new = []
    name_words_list = []  # 所有entity name的单词列表
    for id, l in namesFrame.iloc[:].values:
        entity_name_new.append((id, ' '.join(l)))
        name_words_list += l
    name_words_list = list(set(name_words_list))  # 所有嵌入的单词
    print('names_word:', len(name_words_list))  # 单词数量

    # 嵌入向量
    word_embed_dict, unlisted_words = get_name_word_embed(name_words_list)  # 获得单词列表，已有嵌入结果和未嵌入列表
    print('names word embed:', len(word_embed_dict))  # 单词数量
    unlisted_words_embed = generate_word2vec_by_character_embedding(unlisted_words)  # 未嵌入单词，用字符嵌入
    print('names char embed:', len(unlisted_words_embed))  # 字符数量
    word_embed_dict.update(unlisted_words_embed)   # 合并,
    print('word_embed_dict(sum of up):', len(word_embed_dict))  # 总数量

    names_word_em = np.zeros([len(name_words_list) + 1, 300])  # ndarray(W+1, 300)，默认0向量，在最后增加一个默认零向量
    names_word_em = np.stack(names_word_em, axis=0).astype(np.float64)  # 将list转换为float向量
    count = 0
    for word_id in range(len(name_words_list)):
        if name_words_list[word_id] in word_embed_dict:
            names_word_em[word_id] = word_embed_dict[name_words_list[word_id]]  # 取单词的索引，取单词嵌入
            count += 1
    print('count:', count)  # 单词数量

    names_words_se = pd.Series(name_words_list)
    names_words_se = pd.Series(names_words_se.index, names_words_se.values)

    def lookup_and_padding(x):
        default_length = 4
        ids = list(names_words_se.loc[x].values) + [names_words_se.iloc[-1], ] * default_length  # 将每一行的单词编号，并加入未标记编号
        return ids[:default_length]

    namesFrame.iloc[:, 1] = namesFrame.iloc[:, 1].apply(lookup_and_padding)

    # entity-desc-embedding dataframe
    un_logged_id = len(names_words_se)
    e_desc_input = pd.DataFrame(np.repeat([[un_logged_id, ] * 4], names_len, axis=0),
                                range(names_len))  # (999994,4):ndarray[999994 999994 999994 999994]
    e_desc_input.iloc[namesFrame.iloc[:, 0].values] = np.stack(namesFrame.iloc[:, 1].values)
    # iloc函数为Selection by Position，即按位置选择数据

    entity_embeds = names_word_em[e_desc_input.values]
    entity_embeds = np.sum(entity_embeds, axis=1)
    return entity_name_new, entity_embeds


def get_entity_embed(ent_id2value_list):
    '''    '''
    print('\nget entity embedding')
    start = time.time()

    new_ent_list = list()  # 存储两个kg所有的id-name对应关系,dict(30000)
    for (e_id, e_name) in ent_id2value_list:
        e_name = e_name.split('/')[-1]
        new_ent_list.append((e_id, e_name))

    #aa = np.array(new_ent_list)
    clean_entity_list, name_embeds = get_word_embed(new_ent_list)  # 获得所有单词的嵌入向量
    print('entity_embed_mat:', len(name_embeds))

    print('generating costs time: {:.4f}s'.format(time.time() - start))
    return clean_entity_list, name_embeds


# 公共，属性值/实体的单词嵌入
def get_name_word_embed(names_words):
    # /wiki-news-300d-1M.vec'  wiki-news-300d-1M.pkl
    file_path = '../../datasets2023/'
    if os.path.exists(file_path + 'wiki-news-300d-1M.vec'):
        # 嵌入向量
        listed_words_dict = {}
        print('load word embedding')
        with open(file_path + 'wiki-news-300d-1M.vec', 'r', encoding='utf-8') as f:
            w = f.readlines()
            w = pd.Series(w[1:])  # SERICES:(999994,)
        begin, eachnum = 0, 200000
        allword_sum = len(w)
        while begin < allword_sum:
            end = begin + eachnum if begin + eachnum < allword_sum else allword_sum
            we = w[begin:end].str.split(' ')
            word = we.apply(lambda x: x[0])  # 所有嵌入的单词
            w_em = we.apply(lambda x: x[1:])  # 嵌入向量
            we_dict = dict(zip(word, w_em))
            for v in names_words:
                if v in we_dict:
                    listed_words_dict[v] = we_dict[v]  # 单词嵌入
            begin = end
            print(end, end=',')
    else:
        listed_words_dict = autil.fileUtil.loadpickle(file_path + 'wiki-news-300d-1M.pkl')
    unlisted_words = []
    for w in names_words:
        if w not in listed_words_dict:
            unlisted_words.append(w)  # 无单词嵌入，字符嵌入

    return listed_words_dict, unlisted_words  # 单词嵌入，字符嵌入


def generate_word2vec_by_character_embedding(word_list, vector_dimension=300):
    ''' 字符嵌入 '''
    character_vectors = {}
    alphabet = ''
    ch_num = {}  # 字符频率
    for word in word_list:
        for ch in word:
            n = 1
            if ch in ch_num:
                n += ch_num[ch]
            ch_num[ch] = n
    ch_num = sorted(ch_num.items(), key=lambda x: x[1], reverse=True)
    ch_sum = sum([n for (_, n) in ch_num])
    for i in range(len(ch_num)):
        if ch_num[i][1] / ch_sum >= 0.0001:
            alphabet += ch_num[i][0]

    char_sequences = [list(word) for word in word_list]
    modelword2vec = Word2Vec(char_sequences, vector_size=vector_dimension, window=5, min_count=1)
    for ch in alphabet:
        #assert ch in model  # 原因是gensim 4 版本与 gensim3使用方法不同造成的
        if (ch in modelword2vec.wv):
            character_vectors[ch] = modelword2vec.wv[ch]

    get_word2vec = {}
    for word in word_list:
        vec = np.zeros(vector_dimension, dtype=np.float32)
        for ch in word:
            if ch in alphabet:
                vec += character_vectors[ch]
        if len(word) != 0:
            get_word2vec[word] = vec / len(word)
        # else:
        #     print(word)
    return get_word2vec


def get_value_embed(value_id_dict, tokens_max_len=10, word2vec_dimension=300):
    '''
    定长10的属性值嵌入
    :param value_id_dict: # (vid, value)
    :param tokens_max_len:
    :param word2vec_dimension:
    :return:
    '''
    print('\nget attribute value embedding')
    start = time.time()

    # value 列表
    attr_value_list = list(set(value_id_dict.values()))  # (vid, value) =>value
    print('attr_value_list:', len(attr_value_list))

    # 拆分属性值
    value_names_words = []  # 所有entity name的单词列表
    for v in attr_value_list:
        value_names_words += v.split(' ')
    value_names_words = list(set(value_names_words))  # 所有嵌入的单词和数字
    print('value_names_words:', len(value_names_words))
    # 属性值嵌入
    word_embed_dict, unlisted_words = get_name_word_embed(value_names_words)  # 已有嵌入结果(单词嵌入)，未嵌入列表（字符嵌入）
    print('names word embed:', len(word_embed_dict))  # 单词数量
    unlisted_words_embed = generate_word2vec_by_character_embedding(unlisted_words)  # 未嵌入单词，用字符嵌入
    print('char word embed:', len(word_embed_dict))  # 字符数量
    word_embed_dict.update(unlisted_words_embed)   # 合并
    print('word_embed_dict(sum of up):', len(word_embed_dict))  # 总数量

    value_embed_mat = np.zeros((len(value_id_dict), word2vec_dimension), dtype=np.float32)
    for (id, value) in value_id_dict.items():
        vectors = np.zeros((tokens_max_len, word2vec_dimension), dtype=np.float32)  # 10*300
        value_words = value.split(' ')
        for i in range(min(tokens_max_len, len(value_words))):
            if value_words[i] in word_embed_dict:
                vectors[i] = word_embed_dict[value_words[i]]   # vectors: value值的嵌入向量列表，(10,300)
        value_embed_mat[int(id)] = np.sum(vectors, axis=0)   # 10个300向量，叠加
    #assert len(value_id_dict) == len(value_embed_mat)
    print('value_embed_mat:', len(value_embed_mat))

    print('generating costs time: {:.4f}s'.format(time.time() - start))
    return value_embed_mat  # (len, 300)
