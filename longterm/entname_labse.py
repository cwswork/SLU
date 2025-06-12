# coding: UTF-8
from os.path import join

import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import random
#import faiss
#from settings import *

from tqdm import tqdm
# using labse
from transformers import *
import torch
import pickle

from autil import fileUtil


class LaBSEEncoder(nn.Module):
    def __init__(self, LaBSE_path, MAX_LEN, device):
        super(LaBSEEncoder, self).__init__()
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained(LaBSE_path, do_lower_case=False)
        self.model = AutoModel.from_pretrained(LaBSE_path).to(self.device)
        self.MAX_LEN = MAX_LEN
    def forward(self, batch):
        sentences = batch
        #  text input must of type `str` (single example), `List[str]` (batch or single pretokenized example) or `List[List[str]]` (batch of pretokenized examples).
        tok_res = self.tokenizer(sentences, add_special_tokens=True, padding='max_length', max_length=self.MAX_LEN)
        input_ids = torch.LongTensor([d[:self.MAX_LEN] for d in tok_res['input_ids']]).to(self.device)
        token_type_ids = torch.LongTensor(tok_res['token_type_ids']).to(self.device)
        attention_mask = torch.LongTensor(tok_res['attention_mask']).to(self.device)
        output = self.model(input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        return F.normalize(output[0][:, 1:-1, :].sum(dim=1))

def load_ent_name(load_file):
    out_dict = {}
    #filename =   'ent_ids_' + str(fileid)
    with open(load_file, encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            line = line[:-1].split('\t')
            id = int(line[0])
            entity = str(line[1])
            entity = entity.split('/')[-1].replace('_', ' ').replace('-', ' ').replace('.', ' ').replace('(', '').replace(')', '')
            out_dict[id] = entity
        fileUtil.save_dict2txt(load_file.replace('ent_ids','new_ent_ids0514'), out_dict, save_kv='kv')
    return out_dict
def get_name_embed(LaBSE_model, out_path):
    print('===', out_path, '===')

    if 'zh' in out_path:
        ids_dict1 = load_ent_name( out_path+ '/cleaned_ent_ids_1_simp') # cleaned_ent_ids_1  ent_ids_1 cleaned_ent_ids_1_simp
    else:
        ids_dict1 = load_ent_name( out_path+ '/ent_ids_1')

    kg1_embed = {}
    for i, (_id, _ent_name) in tqdm(enumerate(ids_dict1.items())):
        emb = LaBSE_model([_ent_name]).cpu().detach().numpy().tolist()
        kg1_embed[int(_id)] = emb
    with open(join(out_path, "LaBSE_emb_1.pkl"), 'wb') as f:
        pickle.dump(kg1_embed, f)

    ids_dict2 = load_ent_name( out_path+ '/ent_ids_2')
    kg2_embed = {}
    for i, (_id, _ent_name) in tqdm(enumerate(ids_dict2.items())):
        emb = LaBSE_model([_ent_name]).cpu().detach().numpy().tolist()
        kg2_embed[int(_id)] = emb
    with open(join(out_path, "LaBSE_emb_2.pkl"), 'wb') as f:
        pickle.dump(kg2_embed, f)

    print('====', out_path , '=====')


if __name__ == "__main__":
    torch.manual_seed(37)
    torch.cuda.manual_seed(37)
    np.random.seed(37)

    MAX_LEN = 130
    device = "cuda:0"
    print('cuda.is_available:', str(torch.cuda.is_available()) , device, '=====')

    LaBSE_path = 'D:\代码备份\setu4993_LaBSE'
    LaBSE_model = LaBSEEncoder(LaBSE_path, MAX_LEN, device).to(device)

    dir_path = 'D:\proj2023\datasets2024'
    print('====', dir_path , '=====')
    #out_path = join(dir_path, 'DBP15K', 'zh_en')
    #out_path = join(dir_path, 'WN31', 'EN_DE_15K_V2')
    out_path = join(dir_path, 'DWY100K', 'dbp_yg') # dbp_wd,dbp_yg

    get_name_embed(LaBSE_model, out_path)

    # subset_list =["zh_en", "ja_en" ] #["zh_en", "ja_en", "fr_en"]
    # for subset in subset_list:
    #     out_path = join(dir_path, 'DBP15K', subset)
    #     get_name_embed(LaBSE_model, out_path)

    # subset_list =['WN31/EN_DE_15K_V1/', 'WN31/EN_DE_15K_V2/', 'WN31/EN_FR_15K_V1/', 'WN31/EN_FR_15K_V2/']
    # for subset in subset_list:
    #     out_path = join(dir_path, subset)
    #     #get_clean_entity(out_path)
    #     get_name_embed(LaBSE_model, out_path)

    # subset_list =['DBpedia1M/EN_DE_1M/', 'DBpedia1M/EN_FR_1M/', 'DWY100K/dbp_wd/', ', DWY100K/dbp_yg/']
    # for subset in subset_list:
    #     out_path = join(dir_path, subset)
    #     #get_clean_entity(out_path)
    #     get_Embedding(LaBSE_model, out_path)
