import torch.nn as nn
import torch
from torch_geometric.nn import global_add_pool, global_max_pool, global_mean_pool
from torch_sparse import spmm
from utils import scipy_to_torch_sparse
from math import sqrt


class DecomLayer(nn.Module):
    def __init__(self, args):
        super(DecomLayer, self).__init__()
        self.attention = args.attention
        self.dim_in = args.hidden_dim
        self.dim_out = args.hidden_dim
        self.num_heads = args.num_heads
        self.norm_fact = 1 / sqrt(self.dim_out // self.num_heads)
        # self.norm_fact = 1 / sqrt(self.dim_out)
        self.linear_q = nn.Linear(self.dim_in, self.dim_out, bias=False)
        self.linear_k = nn.Linear(self.dim_in, self.dim_out, bias=False)
        self.linear_v = nn.Linear(self.dim_in, self.dim_out, bias=False)

    def calculate_att(self, x, batch, batch_size):
        # 多头注意力
        num_heads = self.num_heads
        dim_out = self.dim_out // num_heads  # dim_k of each head

        Q = self.linear_q(x).reshape(3, num_heads, dim_out).transpose(0, 1)  # (nh, n, dk)
        K = self.linear_k(x).reshape(3, num_heads, dim_out).transpose(0, 1)  # (nh, n, dk)
        V = self.linear_v(x).reshape(3, num_heads, dim_out).transpose(0, 1)  # (nh, n, dv)

        dist = torch.matmul(Q, K.transpose(1, 2)) * self.norm_fact  # batch, nh, n, n
        dist = torch.softmax(dist, dim=-1)  # nh, n, n

        att = torch.matmul(dist, V)  # nh, n, dv
        att = att.transpose(0, 1).reshape(3, self.dim_out)  # n, dim_v

        # 单头注意力
        # Q = self.linear_q(x)
        # K = self.linear_k(x)
        # V = self.linear_v(x)
        #
        # dist = (Q @ K.T) * self.norm_fact
        # dist = torch.softmax(dist, dim=-1)
        # att = dist @ V

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
            x_dec = f(coefs, d_index[i][0].to(device))
            # x_dec = torch.cat([global_mean_pool(coefs, d_index[i][0].to(device)), global_max_pool(coefs, d_index[i][0].to(device)), global_add_pool(coefs, d_index[i][0].to(device))], dim=1)
            if self.attention:
                x_dec = self.calculate_att(x_dec, batch, batch_size)

            if i == 0:
                # x_pool = f(coefs, d_index[i][0].to(device)).flatten()
                x_pool = x_dec.flatten()
            else:
                x_pool = torch.vstack((x_pool, x_dec.flatten()))

        return x_pool
