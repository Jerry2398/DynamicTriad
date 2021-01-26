import numpy as np
import torch
import scipy.sparse as sp
import random
from torch.utils.data import Dataset, DataLoader


class My_dataset():
    def __init__(self, path, dataset, graph_size, graph_number):
        assert path == "data/email/", "path error"
        assert dataset in ["email", "email_depart1", "email_depart2", "email_depart3",
                           "email_depart4"], "no such dataset {}".format(dataset)
        self.path = path
        self.dataset = dataset
        self.graph_size = graph_size
        self.graph_number = graph_number

    def load_trans(self):
        adj_dict, delta_node_sets, delta_neighbor_list, node_number = self.load_raw_data(path=self.path, dataset=self.dataset, graph_size=self.graph_size, graph_number=self.graph_number)
        data = {}
        for i in range(self.graph_number):
            data[i] = self.transform_data(adj_dict[i], delta_node_sets[i], delta_neighbor_list[i], node_number)
        return data, adj_dict, node_number

    def load_raw_data(self, path="data/email/", dataset="email", graph_size=0.8, graph_number=5):
        print("loading data from {}{}.txt ...".format(path, dataset))
        data = np.genfromtxt(path + dataset + ".txt", delimiter=' ', dtype=np.long)
        source = data[:, 0]
        target = data[:, 1]
        time = data[:, 2]
        id_set = set(np.sort(np.concatenate((source, target), axis=0)))
        id_dict = {j: i for i, j in enumerate(id_set)}
        source_idx = np.array([id_dict[i] for i in source])
        target_idx = np.array([id_dict[i] for i in target])

        timespan = time[-1] - time[0]
        tmp = time[0] + int(timespan * graph_size)
        stride = int((time[-1] - tmp) / graph_number)
        start_time_list = [time[0] + i * stride for i in range(graph_number)]
        end_time_list = [tmp + i * stride for i in range(graph_number)]

        adj_dict = {}
        delta_node_sets = {}
        delta_neighbor_list = {}

        node_number = len(id_dict)

        for i in range(graph_number):
            delta_neighbor_list[i] = {}
            start_time = start_time_list[i]
            end_time = end_time_list[i]
            idx = np.argwhere((time >= start_time) & (time <= end_time)).squeeze()
            src_nodes = source_idx[idx]
            tgt_nodes = target_idx[idx]
            adj = sp.coo_matrix((np.ones(len(src_nodes)), (src_nodes, tgt_nodes)), shape=(node_number, node_number))
            adj = sp.coo_matrix(adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj))
            adj_dict[i] = adj
            if i < graph_number-1:
                next_end_time = end_time_list[i + 1]
                delta_idx = np.argwhere((time >= end_time) & (time <= next_end_time)).squeeze()
                delta_src_nodes = source_idx[delta_idx]
                delta_tgt_nodes = target_idx[delta_idx]
                delta_src_set = set(delta_src_nodes)
                delta_tgt_set = set(delta_tgt_nodes)
                delta_node_sets[i] = delta_src_set | delta_tgt_set
                for j in range(len(delta_src_nodes)):
                    src = delta_src_nodes[j]
                    tgt = delta_tgt_nodes[j]
                    if src not in delta_neighbor_list[i]:
                        delta_neighbor_list[i][src] = set()
                        delta_neighbor_list[i][src].add(tgt)
                    else:
                        delta_neighbor_list[i][src].add(tgt)
                    if tgt not in delta_neighbor_list[i]:
                        delta_neighbor_list[i][tgt] = set()
                        delta_neighbor_list[i][tgt].add(src)
                    else:
                        delta_neighbor_list[i][tgt].add(src)
            else:
                delta_node_sets[i] = None
                delta_neighbor_list[i] = None

        return adj_dict, delta_node_sets, delta_neighbor_list, node_number

    def transform_data(self, adj, delta_node_sets, delta_neighbor_list, node_number):
        data_node_dict = {}
        # data = []
        adj = adj.toarray()
        for k in range(adj.shape[0]):
            for j in range(len(adj[k])):
                if adj[k][j] == 1:
                    neg_j = random.randint(0, node_number-1)
                    w_ik = -1
                    w_jk = adj[j][k]
                    condition = -1
                    i = -1
                    if delta_node_sets is not None:
                        for i in range(adj.shape[0]):
                            if adj[i][k] > 0 and adj[i][j] == 0:
                                w_ik = adj[i][k]
                                if i in delta_node_sets and j in delta_neighbor_list[i]:
                                    condition = 1
                                else:
                                    condition = 0
                                # data.append((j, k, neg_j, i, w_ik, w_jk, condition))
                                if j not in data_node_dict:
                                    data_node_dict[j] = []
                                    data_node_dict[j].append((j, k, neg_j, i, w_ik, w_jk, condition))
                                else:
                                    data_node_dict[j].append((j, k, neg_j, i, w_ik, w_jk, condition))
                                if k not in data_node_dict:
                                    data_node_dict[k] = []
                                    data_node_dict[k].append((j, k, neg_j, i, w_ik, w_jk, condition))
                                else:
                                    data_node_dict[k].append((j, k, neg_j, i, w_ik, w_jk, condition))
                        if condition == -1:
                            # data.append((j, k, neg_j, i, w_ik, w_jk, condition))
                            if j not in data_node_dict:
                                data_node_dict[j] = []
                                data_node_dict[j].append((j, k, neg_j, i, w_ik, w_jk, condition))
                            else:
                                data_node_dict[j].append((j, k, neg_j, i, w_ik, w_jk, condition))
                            if k not in data_node_dict:
                                data_node_dict[k] = []
                                data_node_dict[k].append((j, k, neg_j, i, w_ik, w_jk, condition))
                            else:
                                data_node_dict[k].append((j, k, neg_j, i, w_ik, w_jk, condition))
                    else:
                        if j not in data_node_dict:
                            data_node_dict[j] = []
                            data_node_dict[j].append((j, k, neg_j, i, w_ik, w_jk, condition))
                        else:
                            data_node_dict[j].append((j, k, neg_j, i, w_ik, w_jk, condition))
                        if k not in data_node_dict:
                            data_node_dict[k] = []
                            data_node_dict[k].append((j, k, neg_j, i, w_ik, w_jk, condition))
                        else:
                            data_node_dict[k].append((j, k, neg_j, i, w_ik, w_jk, condition))
                        # data.append((j, k, neg_j, i, w_ik, w_jk, condition))
                else:
                    continue
        return data_node_dict

'''
    def transform_data(self, adj, delta_node_sets, delta_neighbor_list, node_number):
        data = []
        adj = adj.toarray()
        for i in adj.shape[0]:
            for j in range(len(adj[i])):
                if adj[i][j] == 0:
                    ks = []
                    wik = []
                    wjk = []
                    for k in range(len(adj.shape[0])):
                        if adj[i][k] > 0 and adj[k][j] > 0:
                            ks.append(k)
                            wik.append(adj[i][k])
                            wjk.append(adj[k][j])
                        neg_j = np.random.randint(0, node_number, size=len(ks))
                    if i in delta_node_sets and j in delta_neighbor_list[i]:
                        conditions = np.ones_like(np.array(ks))
                    else:
                        conditions = np.zeros_like(np.array(ks))
                    data.append((i, j, ks, neg_j, wik, wjk, conditions))
                else:
                    continue
        return data
'''

'''
当前改动：
将每个图数据的序列化的data改为每个图数据结点相关的data list，便于之后训练过程中的采样的的进行。
'''
