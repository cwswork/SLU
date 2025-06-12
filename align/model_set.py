import sys
import numpy as np
import torch
from time import perf_counter # perf_counter() 返回一个CPU级别的精准时间计数值,单位为秒

from align import model_util, model_loss, pre_loadKGs
from align.model_Lcat import LCAT_model
from autil import fileUtil, alignment

class align_set():
    def __init__(self, myconfig):
        super(align_set, self).__init__()
        self.myconfig = myconfig
        self.best_mode_pkl_title = myconfig.Out_Dir + myconfig.time_str
        #  Load KGs data
        self.kgs_data = pre_loadKGs.load_KGs_data(myconfig)
        ## Hyper Parameter ######################

        ## Model and optimizer ######################
        self.graph_model = LCAT_model(self.kgs_data, myconfig).to(myconfig.device) #
        self._graph_model = LCAT_model(self.kgs_data, myconfig).to(myconfig.device)
        self._graph_model.update(self.graph_model)

        self.optimizer = torch.optim.AdamW(self.graph_model.parameters(), lr=myconfig.peak_lr,
                            weight_decay=myconfig.weight_decay)  # 权重衰减（参数L2损失）weight_decay =5e-4
        self.criterion = model_loss.Nag_Loss(self.myconfig)

        myconfig.myprint(self.graph_model)
        myconfig.myprint('total params:' + str(sum(p.numel() for p in self.graph_model.parameters())))
        myconfig.myprint('-' * 20 + '\n')

        self.train_time, self.valid_time = model_util.CostTimeMeter("Train"), model_util.CostTimeMeter("Valid")
        self.test_time_L, self.test_time_R = model_util.CostTimeMeter("Test_Left"), model_util.CostTimeMeter("Test_Right")
        self.bad_counter, self.min_validloss_counter = 0, 0
        self.best_hits1, self.best_epochs = 0, -1
        self.best_test_hits1, self.best_test_epochs, = 0, -1
        self.min_valid_loss = sys.maxsize

    ## model train
    def model_run(self, beg_epochs=0):
        t_begin = perf_counter()
        last_epoch = 0
        # initial（SF） accuracy
        kg1_name_embed, kg2_name_embed = self.kgs_data.initial_name_embed()
        Left_re, Right_re = self.accuracy(kg1_name_embed, kg2_name_embed, isTest=True)
        [hits_all_L, result_str_L, Hits_list_L] = Left_re
        [hits_all_R, result_str_R, Hits_list_R] = Right_re
        self.myconfig.myprint("initial_SF Test_Left: {}".format(result_str_L))
        self.myconfig.myprint("initial_SF Test_Right: {}\n".format(result_str_R))

        # Begin model run
        for epochs_i in range(beg_epochs, self.myconfig.train_epochs):
            ## Train
            if self.runTrain(epochs_i) == False:
                break

            if (epochs_i>=self.myconfig.start_valid) and (epochs_i % self.myconfig.eval_freq == 0):
                break_re = self.runValid(epochs_i)
                if self.myconfig.early_stop and break_re:
                    last_epoch = epochs_i
                    break

            if epochs_i % self.myconfig.test_freq == 0:
                self.runTest(epochs_i)

        # 输出相关数据
        self.save_model(last_epoch, 'last')  # save last_epochs
        self.myconfig.myprint("Optimization Finished!")
        self.myconfig.myprint('Last epoch-{:04d}:'.format(last_epoch))
        self.myconfig.myprint('Best epoch-{:04d}:'.format(self.best_epochs))
        self.myconfig.myprint('Best Test epoch-{:04d}:'.format(self.best_test_epochs)) # Test Del

        self.myconfig.myprint("##########################")
        self.myconfig.myprint(self.train_time.get_avg_loss())
        self.myconfig.myprint(self.valid_time.get_avg_acc())
        self.myconfig.myprint(self.test_time_L.get_avg_acc())
        self.myconfig.myprint(self.test_time_R.get_avg_acc())
        self.myconfig.myprint("##########################")

        # Last Test
        # re = self.runTest(epochs_i=last_epoch)  #  Testing
        # # Best Test Load model
        self.myconfig.myprint('\nBest epoch-{:04d}:'.format(self.best_epochs))
        if last_epoch != self.best_epochs:
            best_savefile = '{}-epochs-{}-{}.pkl'.format(self.best_mode_pkl_title, self.best_epochs, 'best')
            self.myconfig.myprint('Loading file: {} - {}th epoch'.format(best_savefile, self.best_epochs))
            re = self.reRunTest(best_savefile, self.best_epochs)
        else:
            re = ''

        self.myconfig.myprint("\nTotal time elapsed: {:.4f}s,  {:.6f}h".format(perf_counter() - t_begin, (perf_counter() - t_begin)/3600))
        self.myconfig.myprint('\nmodel arguments:' + self.myconfig.all_args)
        return re

    ## 运行每轮训练
    def runTrain(self, epochs_i):
        t_epoch = perf_counter()
        loss_total = 0
        # Model trainning, Forward pass
        self.graph_model.train()

        for (batch_ids, batch_ent_adj) in self.kgs_data.train_set: # tqdm

            self.optimizer.zero_grad()  # 梯度清零
            ent_out1 = self.graph_model.forward(batch_ids, batch_ent_adj, self.myconfig.feature_dropout)
            with torch.no_grad():
                self._graph_model.eval()
                ent_out2 = self._graph_model.forward(batch_ids, batch_ent_adj, 0.3)

            train_loss = self.criterion.nce_loss(ent_out1, ent_out2)

            # Backward and optimize
            if torch.isnan(train_loss):
                return False
            train_loss.backward(retain_graph=True) #  多个loss的自定义loss, 可以重复利用计算图，提高计算效率.
            self.optimizer.step()
            self._graph_model.update(self.graph_model)
            loss_total += train_loss.detach()

        # measure data loading time
        outstr = self.train_time.update_loss(epochs_i, loss_total, perf_counter() - t_epoch)
        self.myconfig.myprint(outstr)

        return True

    # ## 运行每轮验证
    def runValid(self, epochs_i):
        #t_epoch = perf_counter()
        ent_embed = None
        with torch.no_grad():  # 1
            # 2 Forward pass
            self.graph_model.eval()
            # 3 model action
            for batch_id, (batch_ids, batch_ent_adj) in enumerate(self.kgs_data.val_set):

                batch_embed = self.graph_model.forward(batch_ids, batch_ent_adj)
                batch_embed = batch_embed.squeeze().detach()
                if batch_id == 0:
                    if self.myconfig.test_GPU:
                        ent_embed = batch_embed
                    else:
                        ent_embed = batch_embed.cpu().numpy()
                else:
                    if self.myconfig.test_GPU:
                        ent_embed = torch.cat((ent_embed, batch_embed), dim=0)
                    else:
                        batch_embed = batch_embed.cpu().numpy()
                        ent_embed = np.concatenate((ent_embed, batch_embed), axis=0)

        # 4 Accuracy
        t_epoch = perf_counter()
        kg_size = int(len(ent_embed) / 2)
        Left_re, _ = self.accuracy(ent_embed[0:kg_size,:], ent_embed[kg_size:,:], isTest=False)
        [hits_all_L, result_str_L, Hits_list_L] = Left_re
        outstr = self.valid_time.update_acc(epochs_i, hits_all_L, result_str_L, perf_counter()  - t_epoch)
        self.myconfig.myprint(outstr)

        # ********************no early stop********************************************
        break_re = False
        # save best model in valid
        if hits_all_L[0] >= self.best_hits1:
            self.best_hits1 = hits_all_L[0]
            self.best_epochs = epochs_i
            self.bad_counter = 0
            self.save_model(epochs_i, 'best')  # 保存最好的模型
            self.myconfig.myprint('==Valid==Epoch-{:04d}, better result, best_hits1:{:.4f}..'.format(epochs_i, self.best_hits1))
        else:
            # no best, but save model every 10 epochs
            self.save_model(epochs_i, 'eval')

            self.bad_counter += 1
            self.myconfig.myprint('==bad_counter++:' + str(self.bad_counter))
            # bad model, stop train
            if self.bad_counter == self.myconfig.patience:  # patience=20
                self.myconfig.myprint('==bad_counter, stop training.')
                break_re = True

        return break_re #, hits1_L

    ## 运行每轮测试
    def runTest(self, epochs_i, isSave=False):
        ent_embed = None
        with torch.no_grad():# 1
            # 2 Forward pass
            self.graph_model.eval()
            # 3 model action
            for batch_id, (batch_ids, batch_ent_adj) in enumerate(self.kgs_data.test_set):

                batch_embed = self.graph_model.forward(batch_ids, batch_ent_adj)
                batch_embed = batch_embed.squeeze().detach()
                if batch_id == 0:
                    if self.myconfig.test_GPU == 1:
                        ent_embed = batch_embed
                    else:
                        ent_embed = batch_embed.cpu().numpy()
                else:
                    if self.myconfig.test_GPU == 1:
                        ent_embed = torch.cat((ent_embed, batch_embed),dim=0)
                    else:
                        batch_embed = batch_embed.cpu().numpy()
                        ent_embed = np.concatenate((ent_embed, batch_embed), axis=0)

        # 4 Accuracy
        t_epoch = perf_counter()
        k = int(len(ent_embed) / 2)
        Left_re, Right_re = self.accuracy(ent_embed[0:k, :], ent_embed[k:, :], isTest=True)

        # measure data loading time
        [hits_all_L, result_str_L, Hits_list_L] = Left_re # 准确率等
        outstr_L = "==Test==" + self.test_time_L.update_acc(epochs_i, hits_all_L, result_str_L, perf_counter() - t_epoch)
        self.myconfig.myprint(outstr_L)
        [hits_all_R, result_str_R, Hits_list_R] = Right_re
        outstr_R = "==Test==" + self.test_time_R.update_acc(epochs_i, hits_all_R, result_str_R, perf_counter() - t_epoch)
        self.myconfig.myprint(outstr_R)

         ### Test Del
        if hits_all_R[0] >= self.best_test_hits1:  # hits_all_L
            self.best_test_epochs = epochs_i
            self.best_test_hits1 = hits_all_R[0]
            ###
            with open(self.best_mode_pkl_title + '_Result.txt', "a") as ff:
                ff.write(outstr_L +'\n')
                ff.write(outstr_R +'\n-------------------\n')
        ###############
        if isSave: # only reTest
            self.myconfig.myprint('++++++++ Save TEST Result ++++++++')
            self.save_model(epochs_i, 'TestSave')
            # save Test Result：
            save_file = '{}_{}'.format(self.best_mode_pkl_title, epochs_i)
            fileUtil.save_list2txt(save_file + '_Left_hitslist.txt', Hits_list_L)
            fileUtil.save_list2txt(save_file + '_Right_hitslist.txt', Hits_list_R)

        return outstr_L + '\n' + outstr_R

    def save_model(self, better_epochs_i, epochs_name):
        # save model to file ：
        model_savefile = '{}-epochs-{}-{}.pkl'.format(self.best_mode_pkl_title, better_epochs_i, epochs_name)
        model_state = dict()
        model_state['align_layer'] = self.graph_model.state_dict()
        model_state['myconfig'] = self.myconfig
        torch.save(model_state, model_savefile)

    def reRunTrain(self, model_savefile, beg_epochs, is_cuda=False):
        # load model to file
        self.myconfig.myprint('\nLoading file: {} - {}th epoch'.format(model_savefile, beg_epochs))
        if is_cuda:
            checkpoint = torch.load(model_savefile)
        else:
            checkpoint = torch.load(model_savefile, map_location='cpu')  # GPU->CPU
        self.graph_model.load_state_dict(checkpoint['align_layer'])
        self.myconfig = checkpoint['myconfig']
        self.model_run(beg_epochs=beg_epochs)

    def reRunTest(self, model_savefile, epoch_i):
        # load model to file
        if self.myconfig.is_cuda:
            checkpoint = torch.load(model_savefile)
        else:
            checkpoint = torch.load(model_savefile, map_location='cpu')  # GPU->CPU
        self.graph_model.load_state_dict(checkpoint['align_layer'])
        re = self.runTest(epochs_i=epoch_i, isSave=True)  #  Testing
        return re


    ## 计算准确率—— 只计算验证集和测试集
    def accuracy(self, Left_vec, Right_vec, isTest=False):
        with torch.no_grad():
            if isTest:
                Left_re = alignment.get_hits(Left_vec, Right_vec, self.kgs_data.test_link, self.myconfig.top_k, self.myconfig.metric, self.myconfig.bata)
                # From Right
                Right_re = alignment.get_hits(Right_vec, Left_vec, self.kgs_data.test_link[:, [1, 0]],
                   self.myconfig.top_k, self.myconfig.metric, self.myconfig.bata, LeftRight='Right')
            else:# get_hits -> get_hits_simple
                Left_re = alignment.get_hits_simple(Left_vec, Right_vec, self.myconfig.top_k, self.myconfig.metric, self.myconfig.bata)
                Right_re = None

        return Left_re, Right_re

