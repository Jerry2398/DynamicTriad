import numpy as np
import torch
import scipy.sparse as sp

def load_raw_data(self, path="data/email/", dataset="email", graph_size=0.8, graph_number=5):
    assert path == "data/email/", "path error"
    assert dataset in ["email", "email_depart1", "email_depart2", "email_depart3",
                        "email_depart4"], "no such dataset {}".format(dataset)
    data = np.genfromtxt(path + dataset + ".txt", delimiter=' ', dtype=np.long)
    source = data[:, 0]
    target = data[:, 1]
    time = data[:, 2]
    id_set = set(np.sort(np.concatenate((source, target), axis=0)))
    id_dict = {j: i for i, j in enumerate(id_set)}
    source_idx = np.array([id_dict[i] for i in source])
    target_idx = np.array([id_dict[i] for i in target])
    node_number = len(id_dict)

    timespan = time[-1] - time[0]
    tmp = time[0] + int(timespan * graph_size)
    stride = int((time[-1] - tmp) / graph_number)
    start_time_list = [time[0] + i * stride for i in range(graph_number)]
    end_time_list = [tmp + i * stride for i in range(graph_number)]

    # generate E_sh
    E_sh = {}
    for i in range(graph_number):
        start_time = start_time_list[i]
        end_time = end_time_list[i]
        idx = np.argwhere((time >= start_time) & (time <= end_time)).squeeze()
        src_nodes = source_idx[idx]
        tgt_nodes = target_idx[idx]
        tgt_neg_nodes = np.random.randint(0,node_number, size=tgt_nodes.shape)
        E_sh[i] = zip(src_nodes, tgt_nodes, tgt_neg_nodes)

    # generate E_tr
    E_tr = {}
    for i in range(graph_number-1):


