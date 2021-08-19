import torch.nn as nn
import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_sparse import spmm
from utils import scipy_to_torch_sparse
import numpy as np


class MultiHeadAttention(nn.Module):
    def __init__(self, args):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = args.num_heads
        self.dim_in = args.hidden_dim
        self.dim_out = args.hidden_dim

        self.W_Q = nn.Linear(self.dim_in, self.num_heads * self.dim_out, bias=False)
        self.W_K = nn.Linear(self.dim_in, self.num_heads * self.dim_out, bias=False)
        self.W_V = nn.Linear(self.dim_in, self.num_heads * self.dim_out, bias=False)

        self.linear = nn.Linear(self.num_heads * self.dim_out, self.dim_out, bias=False)
        self.layer_norm = nn.LayerNorm(self.dim_out)
        self.dropout = nn.Dropout(args.dropout)

    def forward(self, x):
        residual = x
        x_size = x.size(0)

        Q = self.W_Q(x).view(x_size, self.num_heads, self.dim_out).transpose(0, 1)
        K = self.W_K(x).view(x_size, self.num_heads, self.dim_out).transpose(0, 1)
        V = self.W_V(x).view(x_size, self.num_heads, self.dim_out).transpose(0, 1)

        scores = torch.matmul(Q, K.transpose(1, 2)) / np.sqrt(self.dim_out)
        scores = torch.softmax(scores, dim=-1)
        attention = torch.matmul(scores, V)
        attention = attention.transpose(0, 1).contiguous().view(x_size, self.num_heads * self.dim_out)
        output = self.linear(attention)
        output = self.dropout(output)

        return self.layer_norm(output + residual)


class MultiScaleLayer(nn.Module):
    def __init__(self, args):
        super(MultiScaleLayer, self).__init__()
        self.device = args.device
        self.ret = args.attention
        self.attention = MultiHeadAttention(args)

    def forward(self, x, batch, batch_size, d_list, d_index):
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
        for i in range(batch_size):
            # extract the i-th graph
            bi = (batch == i)
            index, value, m, n = scipy_to_torch_sparse(d_list[i][0], device=self.device)
            coefs = spmm(index, value, m, n, x[bi, :])
            x_dec = global_add_pool(coefs, d_index[i][0].to(self.device))
            if self.ret:
                x_dec = self.attention(x_dec)

            if i == 0:
                x_pool = x_dec.flatten()
            else:
                x_pool = torch.vstack((x_pool, x_dec.flatten()))

        return x_pool
