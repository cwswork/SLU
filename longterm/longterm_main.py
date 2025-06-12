import argparse
import pickle
import random
from os.path import join
from time import perf_counter

import numpy as np
import networkx as nx
import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
import load_graph
import json

from autil import fileUtil
from entname_labse import LaBSEEncoder


def parse_args():
	'''
	Parses the longterm arguments.
	'''
	parser = argparse.ArgumentParser(description="Run longterm.")

	parser.add_argument('--dimensions', type=int, default=128,
	                    help='Number of dimensions. Default is 128.') # 300

	parser.add_argument('--walk-length', type=int, default=80,
	                    help='Length of walk per source. Default is 80.')

	parser.add_argument('--num-walks', type=int, default=10,
	                    help='Number of walks per source. Default is 10.')

	parser.add_argument('--window-size', type=int, default=10,
                    	help='Context size for optimization. Default is 10.')

	parser.add_argument('--iter', default=5, type=int,
                      help='Number of epochs in SGD')

	parser.add_argument('--workers', type=int, default=12,
	                    help='Number of parallel workers. Default is 8.')

	parser.add_argument('--p', type=float, default=1e-100,
	                    help='Return hyperparameter. Default is 1.')

	parser.add_argument('--q', type=float, default=1,
	                    help='Inout hyperparameter. Default is 1.')

	parser.add_argument('--weighted', dest='weighted', action='store_true',
	                    help='Boolean specifying (un)weighted. Default is unweighted.')
	parser.add_argument('--unweighted', dest='unweighted', action='store_false')
	parser.set_defaults(weighted=False)

	parser.add_argument('--directed', dest='directed', action='store_true',
	                    help='Graph is (un)directed. Default is undirected.')
	parser.add_argument('--undirected', dest='undirected', action='store_false')
	parser.set_defaults(directed=False)

	parser.add_argument('--node2rel', default='none',
                      help='for[h,r,t], node2rel is a dict (h+t)->r')

	return parser.parse_args()

def load_clean_dict(file, sep='\t'):
	# 实体，已编号  ent_ids_1、ent_ids_2
	print('loading ids_dict file ' + file)
	id2dict = dict()
	with open(file, encoding='utf-8') as f:
		for line in f:
			th = line[:-1].split(sep)
			id = int(th[0])
			name = th[1].split('/')[-1]
			name = name.replace('-', ' ').replace('_', ' ').replace('(','').replace(')','')
			id2dict[id] = name  # (eid:name)

	return id2dict

def get_rels(data_dir):
	# rel
	kg1_ent_dict = load_clean_dict(data_dir + 'ent_ids_1')
	kg2_ent_dict = load_clean_dict(data_dir + 'ent_ids_2')
	#ent_ids = kg1_ent_ids + kg2_ent_ids
	KG_E = len(kg1_ent_dict) + len(kg2_ent_dict)

	kg1_rel_dict = load_clean_dict(join(data_dir, 'rel_ids_1'))
	kg2_rel_dict = load_clean_dict(join(data_dir, 'rel_ids_2'))
	KG_R = len(kg1_rel_dict) + len(kg2_rel_dict)

	rel_triples1 = fileUtil.load_triples_id(join(data_dir, 'triples_1'))
	rel_triples2 = fileUtil.load_triples_id(join(data_dir, 'triples_2'))
	rel_triples = rel_triples1+rel_triples2

	print("Num of KG1 entitys:", len(kg1_ent_dict))
	print("Num of KG2 entitys:", len(kg2_ent_dict))
	print("Num of KG1 relations:", len(kg1_rel_dict))
	print("Num of KG2 relations:", len(kg2_rel_dict))
	print("Num of KGs rel triples:", len(rel_triples))
	print("Num of all entitys:", KG_E)

	ent_dict = kg1_ent_dict
	ent_dict.update((kg2_ent_dict))
	rel_dict = kg1_rel_dict
	rel_dict.update((kg2_rel_dict))

	alignRel_dict = fileUtil.loadpickle("{}/path2/alignRel_dict".format(data_dir))
	for r1, r2 in alignRel_dict.items():
		rel_dict[r2] = rel_dict[r1] # 统一两个对齐 rel的名称, 统一为r1

	print("align rel:", len(alignRel_dict))

	new_reltriples = []
	node2rel = {}
	for h, r, t in rel_triples:
		new_reltriples.append((h, r, t))

		h, t = str(h), str(t)
		node2rel[h + '+' + t] = r
		node2rel[t + '+' + h] = r

	return KG_E, KG_R, ent_dict, rel_dict, new_reltriples,node2rel


def learn_embeddings_LaBSE(KG_E, deepwalks_dict, args):
	'''
	Learn embeddings by optimizing the Skipgram objective using SGD.
	'''
	MAX_LEN = 512  #	11*10*2 = 220
	device = args.device  #"cuda:0"
	print('cuda.is_available:', str(torch.cuda.is_available()), device, '=====')

	LaBSE_model = LaBSEEncoder(args.LaBSE_path, MAX_LEN, device).to(device)

	kg_embed = []
	for i_id in tqdm(range(0, KG_E)): # 一个实体，有多条walk
		walk_list = []
		assert args.num_walks == len(deepwalks_dict[i_id])
		for item in deepwalks_dict[i_id]:
			one_walk = ' '.join(item)
			walk_list.append(one_walk) # (10)

		emb_list = LaBSE_model(walk_list).cpu().detach() # .numpy().tolist()
		emb = emb_list.mean(dim=0) # (10,d) ->(d)
		kg_embed.append(emb)
	kg_embed = torch.FloatTensor(np.array(kg_embed))
	with open(args.outputLaBSE, 'wb') as f:
		pickle.dump(kg_embed, f)


# def learn_embeddings(walks_list):
# 	'''
# 	Learn embeddings by optimizing the Skipgram objective using SGD.
# 	'''
# 	walks_list2 = [list(map(str, walk)) for walk in walks_list]
#
# 	model = Word2Vec(walks_list2, vector_size=args.dimensions, window=args.window_size, min_count=0, sg=1,
# 					 workers=args.workers, epochs=args.iter)  # epochs
# 	model.wv.save_word2vec_format(args.outputWord2vec)


def get_deepwalk(args):
	KG_E, KG_R, ent_dict, rel_dict, new_reltriples, node2rel = get_rels(args.data_dir)
	### Pipeline for representational learning for all nodes in a graph.
	walkfile = args.data_dir + 'path2/deepwalk.data'
	with open(walkfile, 'w', encoding='utf-8') as fwrite:
		for h, r, t in new_reltriples:
			fwrite.write(str(h) + ' ' + str(t) + '\n')
	nx_G = load_graph.read_graph(walkfile, args.weighted, args.directed)
	G = load_graph.Graph(nx_G, args.directed, args.p, args.q)
	G.preprocess_transition_probs()
	deepwalks = G.simulate_walks(args.num_walks, args.walk_length)
	###
	print("Insert Relation...")
	deepwalks_dict = {}
	for line in deepwalks:
		# first_name = ent_dict[int(line[0])]
		ent_id = line[0]
		new_line = [ent_dict[ent_id]]
		for i in range(len(line) - 1):
			h, t = line[i], line[i + 1]
			h_name, t_name = ent_dict[h], ent_dict[t]
			r = node2rel[str(h) + '+' + str(t)]
			r_name = rel_dict[r]
			new_line += [r_name, t_name]

		ent_id = int(ent_id)
		if ent_id in deepwalks_dict:
			deepwalks_dict[ent_id] = deepwalks_dict[ent_id] + [new_line]
		else:
			deepwalks_dict[ent_id] = [new_line]

	print('walk_count:', len(deepwalks))
	print('deepwalks_dict:', len(deepwalks_dict))

	### 合并同个节点的多条walk，
	no_deepwalk = 0
	for _id in range(KG_E):  # 一个实体，有多条walk
		if _id not in deepwalks_dict:
			print('----0', _id)
			deepwalks_dict[_id] = [ ent_dict[_id] for i in range(10)]
		else:
			if args.num_walks != len(deepwalks_dict[_id]):
				no_deepwalk += 1
	print('no_deepwalk:', no_deepwalk)
	###
	fileUtil.savepickle("{}path2/deepwalks_dict".format(args.data_dir), [KG_E, deepwalks_dict])
	return KG_E, deepwalks_dict


def main_run(data_dir):
	args = parse_args()
	args.data_dir = data_dir
	args.outputLaBSE = args.data_dir + 'path2/longterm_LaBSE.emb'

	args.device = "cuda:0"  # 'cpu'
	args.LaBSE_path = 'D:/代码备份/setu4993_LaBSE/'
	args.q = 0.7
	args.num_walks = 10
	args.walk_length = 5

	####
	KG_E, deepwalks_dict = get_deepwalk(args)
	# learn_embeddings(walks_list)
	learn_embeddings_LaBSE(KG_E, deepwalks_dict, args)

if __name__ == "__main__":
	seed = 26
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed_all(seed)
	t_begin = perf_counter()

	# d = 'DBP15K/zh_en/'  # fr_en, ja_en, zh_en
	# d = 'WN31/EN_FR_15K_V1/' # EN_DE_15K_V1、EN_FR_15K_V1
	# d = 'DWY100K/dbp_wd/' # DWY100K/dbp_wd, DWY100K/dbp_yg
	# main_run('../../datasets2024/' + d)

	# data_list = ['DBP15K/zh_en/', 'WN31/EN_DE_15K_V1/', 'WN31/EN_DE_15K_V2/', 'WN31/EN_FR_15K_V1/', 'WN31/EN_FR_15K_V2/', ]
	data_list = ['DWY100K/dbp_wd/','DWY100K/dbp_yg/']
	for d in data_list:
		main_run('../../datasets2024/' + d)

	print("\nTotal time : {:.4f}s".format(perf_counter() - t_begin))
