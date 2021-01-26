import argparse
import random
import numpy as np
import torch
import torch.optim as optim

from model import DyTraid
from utsils import My_dataset

parser = argparse.ArgumentParser()
parser.add_argument("--cuda", action="store", default=False, type=bool, help="use cuda or not")
parser.add_argument("--path", action="store", default="data/email/", type=str, help="path of data")
parser.add_argument("--dataset", action="store", default="email", type=str, help="name of data")
parser.add_argument("--graph_size", action="store", default=0.8, type=float, help="graph size")
parser.add_argument("--graph_number", action="store", default=5, type=int, help="number of graphs")
parser.add_argument("--seed", action="store", default=72, type=int, help="random seed")
parser.add_argument("--embedding_dim", action="store", default=16, type=int, help="embedding dimension")
parser.add_argument("--lr", action="store", default=0.001, type=float, help="learning rate")
parser.add_argument("--weight_decay", action="store", default=1e-5, type=float, help="weight_decay")
parser.add_argument("--beta0", action="store", default=0.1, type=float, help="hyperparameter: beta 0")
parser.add_argument("--beta1", action="store", default=0.1, type=float, help="hyperparameter: beta 1")
parser.add_argument("--batch_size", action="store", default=256, type=int, help="batch size")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)
torch.manual_seed(args.seed)

dataloader = My_dataset(args.path, args.dataset, args.graph_size, args.graph_number)
# 训练数据，格式为dict，每个key代表图数据编号，item是一个dict，这个dict的key是结点的编号，内容是结点的交互信息
training_data, adj_dict, node_number = dataloader.load_trans()

args.node_number = node_number
model = DyTraid(args.graph_number, args.node_number, args.embedding_dim, args.beta0, args.beta1)
opt = optim.SGD(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)


def train_model(timestamp, data, adj, idx):
    model.train()
    loss = model.fit_transform(timestamp, data, adj, idx)
    opt.zero_grad()
    loss.backward()
    opt.step()
    return loss.item()


def sample_data(timestamp, node_id):
    assert  node_id in training_data[t], "no such node {} in {}th graph data".format(node_id, timestamp)
    data = training_data[t][node_id]
    if len(data) <= args.batch_size:
        weights = torch.ones(len(data))
        idx = torch.multinomial(weights, len(data), replacement=False)
        return idx
    else:
        weights = torch.ones(len(data))
        idx = torch.multinomial(weights, args.batch_size, replacement=False)
        return idx

print("training begins")
for node in range(args.node_number):
    for t in range(args.graph_number):
        if node in training_data[t]:
            idx = sample_data(t, node)
            loss = train_model(t, training_data[t][node], adj_dict[t], idx)
            print("loss is ", loss)


'''
random sample:
idx = torch.multinomimal(nodes_weight, sample_size, replacement=True)

'''

