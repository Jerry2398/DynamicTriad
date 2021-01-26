import torch
import torch.nn as nn
from tqdm import *
import torch.nn.functional as F
import scipy.sparse as sp


class DyTraid(nn.Module):
    def __init__(self, graph_number, node_number, embed_dim, beta0, beta1):
        super(DyTraid, self).__init__()
        self.node_number = node_number
        self.graph_number = graph_number
        self.embed_dim = embed_dim
        self.beta0 = beta0
        self.beta1 = beta1
        self.embeddings = nn.ParameterList([nn.Parameter(torch.FloatTensor(node_number, embed_dim)) for i in range(graph_number)])
        self.theta = nn.Parameter(torch.FloatTensor(node_number, embed_dim))
        self.margin = 1
        self.init_parameters()

    def init_parameters(self):
        for i, item in enumerate(self.parameters()):
            torch.nn.init.xavier_uniform(item, gain=1)

    def get_embeddings(self, i):
        return self.embeddings[i]

    def calculate_C(self, embeddings, i, j, k, adj):
        ks = []
        for l in range(adj.shape[0]):
            if adj[i][l] > 0 and adj[j][l] > 0:
                ks.append(l)
        C0 = self.calculate_P(embeddings, i, j, k, adj)
        C1 = torch.tensor([1 - self.calculate_P(embeddings, i, j, k, adj) for k in ks]).squeeze()
        C1 = 1 - torch.prod(C1)
        eps = 1e-6
        C = 1 - torch.div(C0, C1 + eps)
        return C

    def calculate_P(self, embeddings, i, j, k, adj):
        x = self.calculate_x(embeddings, i, j, k, adj)
        theta = self.theta[k]
        power = torch.sum((-1) * torch.mul(theta, x))
        if power > 100:
            return torch.zeros_like(power)
        else:
            return torch.div(1.0, (1 + torch.exp(power)))

    def calculate_x(self, embeddings, i, j, k, adj):
        w_ik = adj[i][k]
        w_jk = adj[j][k]
        u_k = embeddings[k]
        u_i = embeddings[i]
        u_j = embeddings[j]
        x = w_ik * (u_k - u_i) + w_jk * (u_k - u_j)
        return x

    def calculate_logp(self, x, k):
        dot_tmp = torch.sum((-1) * torch.mul(self.theta[k], x))
        exp_tmp = torch.exp(dot_tmp)
        return torch.log(1 + exp_tmp)

    def L_tr(self, embeddings, j, k, i, adj, condition):
        if condition == -1:
            return 0
        C = self.calculate_C(embeddings, i, j, k, adj)
        x = self.calculate_x(embeddings, i, j, k, adj)
        log_p = self.calculate_logp(x, k)
        dot = torch.sum(torch.mul(self.theta[k], x))
        if condition == 1:
            vest_term = C
        else:
            vest_term = dot
        L_tr = torch.sum(log_p + vest_term)
        return L_tr

    def L_sh(self, embeddings, j, k, neg_j):
        u_j = embeddings[j]
        u_k = embeddings[k]
        u_neg_j = embeddings[neg_j]
        pos = torch.sum(torch.mul(u_j - u_k, u_j - u_k))
        neg = torch.sum(torch.mul(u_neg_j - u_k, u_neg_j - u_k))
        # torch.sum(torch.where(pos - neg + self.margin > 0, pos - neg + self.margin, 0))
        if pos - neg + self.margin > 0:
            L_sh = pos - neg + self.margin
        else:
            L_sh = 0
        return L_sh

    def L_smooth(self, embeddings, last_embeddings):
        L_smooth = torch.sum(torch.mul(embeddings - last_embeddings, embeddings - last_embeddings))
        return L_smooth

    def fit_transform(self, timestamp, data, adj, idx):
        embeddings = self.get_embeddings(timestamp)
        print(embeddings)
        print(embeddings.shape)
        L_sh = 0
        L_smooth = 0
        L_tr = 0
        adj = adj.toarray()
        for i in tqdm(range(len(idx))):
            index = idx[i]
            j, k, neg_j, i, w_ik, w_jk, condition = data[index]
            L_sh = L_sh + self.L_sh(embeddings, j, k, neg_j)  # 网络loss
            if timestamp == 0:
                L_tr = L_tr + self.L_tr(embeddings, j, k, i, adj, condition)  # tr loss
            elif timestamp == self.graph_number:
                last_embeddings = self.get_embeddings(timestamp - 1)
                L_smooth = L_smooth + self.L_smooth(embeddings, last_embeddings)  # 时序loss
            else:
                last_embeddings = self.get_embeddings(timestamp - 1)
                L_smooth = L_smooth + self.L_smooth(embeddings, last_embeddings)        #时序loss
                L_tr = L_tr + self.L_tr(embeddings, j, k, i, adj, condition)    #tr loss
        loss = L_sh + self.beta0 * L_tr + self.beta1 * L_smooth
        return loss
