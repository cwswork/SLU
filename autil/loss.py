import torch
import torch.nn as nn
import numpy as np
import scipy.stats as stats
import torch.nn.functional as F
from model.policy import *



class CL_Loss(nn.Module):

    def __init__(self, args, device):
        super(CL_Loss, self).__init__()
        self.device = device
        self.t = args.t
        self.beta = args.beta
        self.tau_plus = args.tau_plus
        self.batch_size = args.batch_size
        self.loss_choice = args.loss_choice

        self.CrossEntropyLoss = nn.CrossEntropyLoss()
        # self.beta_policy = AdaptiveThresholdPolicy(
        #     lower_is_better=False,
        #     init_thres=self.beta,
        #     min_thres=0.1,
        #     max_thres=1.0,
        #     delta=0.01,
        #     window=5,
        # )

    def forward(self, pos_1, pos_2,  epoch):
        if self.loss_choice == 'infonce':
            loss = self.semi_loss(pos_1, pos_2)
        elif self.loss_choice == 'sim':
            loss = self.simCLR_loss(pos_1, pos_2)
        elif self.loss_choice == 'mixh':
            loss = self.mixh_loss(pos_1, pos_2)
        # elif self.loss_choice == 'hcl':
        #     loss = self.hcl_loss(pos_1, pos_2)
        elif self.loss_choice == 'ce': # 交叉熵损失可以采用“sigmoid+BCE”或是“softmax+CE”
            loss = self.ce_loss(pos_1)

        return loss


    def find_hard_negatives(self, logits):
        """Finds the top n_hard hardest negatives in the queue for the query.

        Args:
            logits (torch.tensor)[batch_size, len_queue]: outputPath dot product negative logits.

        Returns:
            torch.tensor[batch_size, n_hard]]: Indices in the queue.
        """
        # logits -> [batch_size, len_queue]
        _, idxs_hard = torch.topk(
            logits.clone().detach(), k=1, dim=-1, sorted=False)
        # idxs_hard -> [batch_size, n_hard]

        return idxs_hard

    def mixup(self, input, alpha, share_lam=False):
        if not isinstance(alpha, (list, tuple)):
            alpha = [alpha, alpha]
        beta = torch.distributions.beta.Beta(*alpha)
        randind = torch.randperm(input.shape[0], device=input.device)
        if share_lam:
            lam = beta.sample().to(device=input.device)
            lam = torch.max(lam, 1. - lam)
            lam_expanded = lam
        else:
            lam = beta.sample([input.shape[0]]).to(device=input.device)
            lam = torch.max(lam, 1. - lam)
            lam_expanded = lam.view([-1] + [1] * (input.dim() - 1))
        outputPath = lam_expanded * input + (1. - lam_expanded) * input[randind]
        lam = torch.concat((lam, lam), dim=0)
        return outputPath, randind, lam


    def mixh_loss(self, q, k, alpha=1.0):
        mix_q, labels_aux, lam = self.mixup(q, alpha)
        loss = self.nce_loss(q, k)
        return lam * self.nce_loss(q, k) + (1. - lam) * self.nce_loss(mix_q, k)


    def hcl_loss(self, z1: torch.Tensor, z2: torch.Tensor, hard_neg):
        f = lambda x: torch.exp(x / self.t)
        refl_sim = self.sim(z1, z1)
        between_sim = self.sim(z1, z2)
        neg_sim = self.sim(z1, hard_neg)
        refl_sim = f(refl_sim)
        between_sim = f(between_sim)
        neg_sim = f(neg_sim)
        return (-torch.log(between_sim.diag() / (neg_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag()))).mean()

    # def gcl_loss(self, pos_1, pos_2, epoch, bmm_model):
    #     if epoch < self.epoch_start:
    #         loss = self.semi_loss(pos_1, pos_2)
    #     else:
    #         loss = self.semi_loss_bmm(pos_1, pos_2, epoch, bmm_model, fit=True)
    #
    #     return loss

    def sim(self, z1: torch.Tensor, z2: torch.Tensor):
        z1 = F.normalize(z1)
        z2 = F.normalize(z2)
        return torch.mm(z1, z2.t())

    def cosine(self, x1, x2):
        return torch.cosine_similarity(x1, x2, dim=1, eps=1e-08)

    def ce_loss(self, pos_1):
        ''' 二分类loss'''
        #contiguous方法改变了多维数组在内存中的存储顺序，以便配合view方法使用
        logits = torch.mm(pos_1, pos_1.t().contiguous() / self.t)  # (E,E)
        # torch.ones_like函数和torch.zeros_like函数的基本功能是根据给定张量，生成与其形状相同的全1张量或全0张量：
        mask = (torch.ones_like(logits) - torch.eye(self.batch_size, device=logits.device)).bool()  # (E,E) 非斜角都是True
        logits = logits.masked_select(mask).view(self.batch_size, -1) # 只取True的数据
        labels = torch.zeros([self.batch_size]).to(self.device).long()  # 全0？？
        loss = self.CrossEntropyLoss(logits, labels)
        return loss
    def semi_loss(self, z1: torch.Tensor, z2: torch.Tensor): # semi_loss
        f = lambda x: torch.exp(x / self.t)
        refl_sim = self.sim(z1, z1)  # (E1, d) (d,E1)=>(E1,E1)
        between_sim = self.sim(z1, z2) # (E1, d) (d,E2)=>(E1,E2)
        refl_sim = f(refl_sim)
        between_sim = f(between_sim)
        return (-torch.log(between_sim.diag() / (between_sim.sum(1) + refl_sim.sum(1) - refl_sim.diag()))).mean()


    def simCLR_loss(self,  pos_1, pos_2):
        ''' 除了自身，batch内的其他节点都是负样本 '''
        out = torch.cat([pos_1, pos_2], dim=0)  # (2B, d)
        # [2*B, 2*B]
        sim_matrix = torch.exp(torch.mm(out, out.t().contiguous()) / self.t)  # (2B, 2B)
        mask = (torch.ones_like(sim_matrix) - torch.eye(2 * self.batch_size, device=sim_matrix.device)).bool() # (E,E) 非斜角都是True
        # [2*B, 2*B-1]
        sim_matrix = sim_matrix.masked_select(mask).view(2 * self.batch_size, -1) # 只取True的数据，非斜角的所有元素，叠加
        # 同一图片内部不同patch也是负样本，除了自己和自己

        # compute loss
        pos_sim = torch.exp(torch.sum(pos_1 * pos_2, dim=-1) / self.t)  ## (2B, 1)
        # [2*B]
        pos_sim = torch.cat([pos_sim, pos_sim], dim=0)
        loss = - torch.log(pos_sim / sim_matrix.sum(dim=-1))

        # sim_matrix非斜角的所有元素，叠加，是字母
        return loss

    def get_negative_mask(self, batch_size):
        negative_mask = torch.ones((batch_size, 2 * batch_size), dtype=bool)  # (B,2B)
        for i in range(batch_size):
            negative_mask[i, i] = 0
            negative_mask[i, i + batch_size] = 0

        negative_mask = torch.cat((negative_mask, negative_mask), 0)  # (2B,2B)
        return negative_mask

    def get_reliable_mask(self, neg, batch_size):
        reliable_mask = torch.ones((2 * batch_size, neg.shape[1]), dtype=bool)
        _, max_id = torch.max(neg, 1)
        # _, min_id = torch.min(neg, 1)
        max_id = max_id.tolist()
        # min_id = min_id.tolist()
        for i in range(2 * batch_size):
            reliable_mask[i, max_id[i]] = 0
            # reliable_mask[i, min_id[i]] = 0

        return reliable_mask


def weighted_mean(x, w):
    return torch.sum(w * x) / torch.sum(w)


def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar) ** 2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) / x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters,
                 alphas_init,
                 betas_init,
                 weights_init):
        self.alphas = alphas_init
        self.betas = betas_init
        self.weight = weights_init
        self.max_iters = max_iters
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        x_cpu = x.cpu().detach().numpy()
        alpha_cpu = self.alphas.cpu().detach().numpy()
        beta_cpu = self.betas.cpu().detach().numpy()
        return torch.from_numpy(stats.beta.pdf(x_cpu, alpha_cpu[y], beta_cpu[y])).to(x.device)

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return self.weighted_likelihood(x, 0) + self.weighted_likelihood(x, 1)

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r = torch.cat((self.weighted_likelihood(x, 0).view(1, -1), self.weighted_likelihood(x, 1).view(1, -1)), 0)
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(0)
        return r

    def fit(self, x):
        eps = 1e-12
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):
            # E-step
            r = self.responsibilities(x)
            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            if self.betas[1] < 1:
                self.betas[1] = 1.01
            self.weight = r.sum(1)
            self.weight /= self.weight.sum()
        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
