import torch.nn as nn
import torch
from torch_geometric.nn.pool.topk_pool import topk, filter_adj
from torch_geometric.nn import GCNConv
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_sparse import spmm
from utils import scipy_to_torch_sparse
from math import sqrt
import time


class SAGPool(torch.nn.Module):
    def __init__(self, in_channels, ratio=0.8, Conv=GCNConv, non_linearity=torch.tanh):
        super(SAGPool, self).__init__()
        self.in_channels = in_channels
        self.ratio = ratio
        self.score_layer = Conv(in_channels, 1)
        self.non_linearity = non_linearity

    def forward(self, x, edge_index, edge_attr=None, batch=None):
        if batch is None:
            batch = edge_index.new_zeros(x.size(0))
        score = self.score_layer(x, edge_index).squeeze()

        perm = topk(score, self.ratio, batch)
        x = x[perm] * self.non_linearity(score[perm]).view(-1, 1)
        batch = batch[perm]
        edge_index, edge_attr = filter_adj(
            edge_index, edge_attr, perm, num_nodes=score.size(0))

        return x, edge_index, edge_attr, batch, perm


class DecomLayer(nn.Module):
    def __init__(self, args):
        super(DecomLayer, self).__init__()
        self.attention = args.attention
        self.dim_in = args.hidden_dim
        self.dim_out = args.hidden_dim
        self.norm_fact = 1 / sqrt(self.dim_out)

        self.linear_q = nn.Linear(self.dim_in*3, self.dim_out, bias=False)
        self.linear_k = nn.Linear(self.dim_in*3, self.dim_out, bias=False)
        self.linear_v = nn.Linear(self.dim_in*3, self.dim_out, bias=False)

    def calculate_att(self, x):
        Q = self.linear_q(x)  # batch, n, dim_k
        K = self.linear_k(x)  # batch, n, dim_k
        V = self.linear_v(x)  # batch, n, dim_v

        dist = (Q @ K.T) * self.norm_fact
        dist = torch.softmax(dist, dim=-1)
        att = dist @ V

        return att

    def forward(self, x, batch, batch_size, d_list, d_index, aggre_mode='sum'):
        """
           Using Undecimated Framelet Transform.

           :param x: batched hidden representation. shape: [# Node_Sum_Batch, # Hidden Units]
           :param batch: batch index. shape: [# Node_Sum_Batch]
           :param batch_size: integer batch size.
           :param d_list: a list of matrix operators, where each element is a torch sparse tensor stored in a list.
           :param d_index: a list of index tensors, where each element is a torch dense tensor used for aggregation.
           :param aggre_mode: aggregation mode. choices: sum, max, and avg. (default: sum)
           :return: batched vectorial representation for the graphs in the batch.
           """
        if aggre_mode == 'sum':
            f = global_add_pool
        elif aggre_mode == 'avg':
            f = global_mean_pool
        elif aggre_mode == 'max':
            f = global_max_pool
        else:
            raise Exception('aggregation mode is invalid')

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

        for i in range(batch_size):
            # extract the i-th graph
            bi = (batch == i)
            index, value, m, n = scipy_to_torch_sparse(d_list[i][0])
            coefs = spmm(index, value, m, n, x[bi, :])
            # x_dec = f(coefs, d_index[i][0].to(device))
            x_dec = torch.cat([global_mean_pool(coefs, d_index[i][0].to(device)), global_max_pool(coefs, d_index[i][0].to(device)), global_add_pool(coefs, d_index[i][0].to(device))], dim=1)
            if self.attention:
                x_dec = self.calculate_att(x_dec)

            if i == 0:
                # x_pool = f(coefs, d_index[i][0].to(device)).flatten()
                x_pool = x_dec.flatten()
            else:
                x_pool = torch.vstack((x_pool, x_dec.flatten()))

        return x_pool
