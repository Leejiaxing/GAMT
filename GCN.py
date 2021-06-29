import torch
import torch.nn.functional as F
import torch.nn as nn
import numpy as np
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGEConv, GATConv, SAGPooling,EdgePooling,dense_diff_pool
from torch_geometric.utils import to_dense_adj, dense_to_sparse

def mixup_readout(x, y, lam, perm):
    y_a = y
    y_b = y[perm]
    x_mix = lam * x + (1 - lam) * x[perm, :]
    return x_mix, y_a, y_b, lam


class GCN(nn.Module):
    def __init__(self, args):
        super(GCN, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden_dim = args.hidden_dim
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        # self.pooling_ratio = args.pooling_ratio
        # self.sample = args.sample_neighbor
        # self.sparse = args.sparse_attention
        # self.sl = args.structure_learning
        # self.lamb = args.lamb
        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)

        # self.pool1 = SAGPooling(self.hidden_dim, ratio=self.pooling_ratio)
        # self.pool2 = SAGPooling(self.hidden_dim, ratio=self.pooling_ratio)
        # self.pool3 = SAGPooling(self.hidden_dim, ratio=self.pooling_ratio)
        # self.pool1 = HGPSLPool(self.hidden_dim, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        # self.pool2 = HGPSLPool(self.hidden_dim, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)
        # self.pool3 = HGPSLPool(self.hidden_dim, self.pooling_ratio, self.sample, self.sparse, self.sl, self.lamb)

        self.fc1 = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)

        return x

    def forward(self, data, mixup=False, alpha=0.1):
        x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
        edge_attr = None
        adj = to_dense_adj(edge_index, batch)

        if mixup:
            if alpha > 0.0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.0
            perm = torch.randperm(y.size(0)).cuda()
            y_perm = y[perm]

            x = F.relu(self.conv1(x, edge_index, edge_attr))
            # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            # x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
            x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
            x1_mix = lam * x1 + (1 - lam) * x1[perm, :]

            x = F.relu(self.conv2(x, edge_index, edge_attr))
            # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            # x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
            x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
            x2_mix = lam * x2 + (1 - lam) * x2[perm, :]

            x = F.relu(self.conv3(x, edge_index, edge_attr))
            # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            # x, edge_index, edge_attr, batch = self.pool3(x, edge_index, edge_attr, batch)
            x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
            x3_mix = lam * x3 + (1 - lam) * x3[perm, :]
            x = x1_mix + x2_mix + x3_mix
            # x=torch.cat([x1_mix,x2_mix,x3_mix],dim=1)
            x = self.fc_forward(x)

            return x, y_perm, lam
        else:
            x = F.relu(self.conv1(x, edge_index, edge_attr))
            # x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
            # x, edge_index, edge_attr, batch = self.pool1(x, edge_index, edge_attr, batch)
            x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)

            x = F.relu(self.conv2(x, edge_index, edge_attr))
            # x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
            # x, edge_index, edge_attr, batch = self.pool2(x, edge_index, edge_attr, batch)
            x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)

            x = F.relu(self.conv3(x, edge_index, edge_attr))
            # x, edge_index, _, batch, _, _ = self.pool3(x, edge_index, None, batch)
            # x, edge_index, edge_attr, batch = self.pool3(x, edge_index, edge_attr, batch)
            x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)
            x = x1 + x2 + x3
            # x = torch.cat([x1, x2, x3], dim=1)
            x = self.fc_forward(x)

            return x
