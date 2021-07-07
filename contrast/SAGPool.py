import torch
import torch.nn.functional as F
import torch.nn as nn
from torch_geometric.nn import global_mean_pool, global_max_pool, global_add_pool
from torch_geometric.nn import GCNConv, SAGPooling


class SAGPool(nn.Module):
    def __init__(self, args):
        super(SAGPool, self).__init__()
        self.args = args
        self.num_features = args.num_features
        self.hidden_dim = args.hidden_dim
        self.num_classes = args.num_classes
        self.dropout = args.dropout
        self.pooling_ratio = 0.5

        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)

        self.pool1 = SAGPooling(self.hidden_dim, ratio=self.pooling_ratio)
        self.pool2 = SAGPooling(self.hidden_dim, ratio=self.pooling_ratio)
        self.pool3 = SAGPooling(self.hidden_dim, ratio=self.pooling_ratio)

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

    def forward(self, data):
        x, y, edge_index, batch = data.x, data.y, data.edge_index, data.batch
        edge_attr = None

        x = F.relu(self.conv1(x, edge_index, edge_attr))
        x, edge_index, _, batch, _, _ = self.pool1(x, edge_index, None, batch)
        x1 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)

        x = F.relu(self.conv2(x, edge_index, edge_attr))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x2 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)

        x = F.relu(self.conv3(x, edge_index, edge_attr))
        x, edge_index, _, batch, _, _ = self.pool2(x, edge_index, None, batch)
        x3 = torch.cat([global_mean_pool(x, batch), global_max_pool(x, batch), global_add_pool(x, batch)], dim=1)

        x = x1 + x2 + x3
        x = self.fc_forward(x)

        return x
