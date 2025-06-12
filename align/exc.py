import os
import time

from align.model_set import align_set
from align import configUtil
from align.configUtil import configClass



if __name__ == '__main__':
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1' # 这样可以保证报错信息提示的位置是准确的
    # 使用 parse_args() 解析添加的参数
    args = configUtil.parse_args()
    #args.dataset = 'DBP15K/zh_en/' ## fr_en, ja_en, zh_en
    args.dataset = 'WN31/EN_DE_15K_V2/'
    args.device = "0"  # -1

    #args.bata = get_bata(args.dataset)
    args.ent_adj_file = 'path2/ent_adj_new5' # ent_adj_csr ent_adj_new5 path_adj5
    args.longterm_emb = 'path2/longterm_LaBSE.emb'
    args.name_embed = 'LaBSE_emb'  # 默认是LaBSE_emb

    # tt
    #args.gcn_dropout = 0
    args.out_dim = 1536
    args.peak_lr = 3e-5 #5e-5
    #args.mode = 'gat' # lcat gat
    #args.metric = 'CDistance'
    #args.bata = get_bata(args.dataset)
    args.name = '0528path2_nodropout'

    # 模型执行开始
    print('---------------------------------------------')
    myconfig = configClass(args, os.path.realpath(__file__))
    ## 定义模型及运行
    mymodel = align_set(myconfig)
    mymodel.model_run()
    myconfig.myprint("end==" + time.strftime('%Y.%m.%d %H:%M:%S', time.localtime(time.time())))
    print('---------------------------------------------')

################
