import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, JumpingKnowledge
from layers import MultiScaleLayer


class Model(nn.Module):
    def __init__(self, args, r):
        super(Model, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.num_layers = args.num_layers
        self.hidden_dim = args.hidden_dim
        self.scale = (r - 1) * args.Lev + 1

        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.convs = torch.nn.ModuleList()
        for i in range(self.num_layers - 1):
            self.convs.append(GCNConv(self.hidden_dim, self.hidden_dim))

        self.multi_scale = MultiScaleLayer(args).to(args.device)

        self.fc1 = nn.Linear(self.scale * self.hidden_dim, self.hidden_dim)
        self.fc2 = nn.Linear(self.hidden_dim, self.hidden_dim // 2)
        self.fc3 = nn.Linear(self.hidden_dim // 2, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)
        return x

    def forward(self, data, mixup=False, alpha=0.1):
        x, y, edge_index, batch, d, d_index = data.x, data.y, data.edge_index, data.batch, data.d, data.d_index
        batch_size = int(batch.max() + 1)

        if mixup:
            if alpha > 0.0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.0
            perm = torch.randperm(y.size(0), device=self.args.device)
            y_perm = y[perm]

            # 3 convolutional layers
            x = F.relu(self.conv1(x, edge_index))
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))

            # 1 Framelet transform layer
            x = self.multi_scale(x, batch, batch_size, d, d_index)
            x_mix = lam * x + (1 - lam) * x[perm, :]
            x = self.fc_forward(x_mix)

            return x, y_perm, lam
        else:
            # 3 convolutional layers
            x = F.relu(self.conv1(x, edge_index))
            for conv in self.convs:
                x = F.relu(conv(x, edge_index))

            # 1 Framelet transform layer
            x = self.multi_scale(x, batch, batch_size, d, d_index)

            x = self.fc_forward(x)

            return x


class ModelwithJK(nn.Module):
    def __init__(self, args, r):
        super(ModelwithJK, self).__init__()
        self.args = args
        self.dropout = args.dropout
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.hidden_dim = args.hidden_dim

        self.conv1 = GCNConv(self.num_features, self.hidden_dim)
        self.conv2 = GCNConv(self.hidden_dim, self.hidden_dim)
        self.conv3 = GCNConv(self.hidden_dim, self.hidden_dim)

        self.decomposition = MultiScaleLayer(args).to(args.device)
        self.jump = JumpingKnowledge('cat')
        self.fc1 = nn.Linear(((r - 1) * args.Lev + 1) * self.hidden_dim * 3, self.hidden_dim * 3)
        self.fc2 = nn.Linear(self.hidden_dim * 3, self.hidden_dim)
        self.fc3 = nn.Linear(self.hidden_dim, self.num_classes)

    def reset_parameters(self):
        self.jump.reset_parameters()
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        self.conv3.reset_parameters()
        self.fc1.reset_parameters()
        self.fc2.reset_parameters()
        self.fc3.reset_parameters()

    def fc_forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.relu(self.fc2(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = F.log_softmax(self.fc3(x), dim=-1)
        return x

    def forward(self, data, mixup=False, alpha=0.1):
        x, y, edge_index, batch, d, d_index = data.x, data.y, data.edge_index, data.batch, data.d, data.d_index
        batch_size = int(batch.max() + 1)

        if mixup:
            if alpha > 0.0:
                lam = np.random.beta(alpha, alpha)
            else:
                lam = 1.0
            perm = torch.randperm(y.size(0), device=self.args.device)
            y_perm = y[perm]

            # 3 convolutional layers
            x = F.relu(self.conv1(x, edge_index))
            xs = [x]
            x = F.relu(self.conv2(x, edge_index))
            xs += [x]
            x = F.relu(self.conv3(x, edge_index))
            xs += [x]
            x = self.jump(xs)
            # 1 Framelet decomposition layer
            x = self.decomposition(x, batch, batch_size, d, d_index)
            x_mix = lam * x + (1 - lam) * x[perm, :]
            x = self.fc_forward(x_mix)

            return x, y_perm, lam
        else:
            # 3 convolutional layers
            x = F.relu(self.conv1(x, edge_index))
            xs = [x]
            x = F.relu(self.conv2(x, edge_index))
            xs += [x]
            x = F.relu(self.conv3(x, edge_index))
            xs += [x]
            x = self.jump(xs)

            # 1 Framelet decomposition layer
            x = self.decomposition(x, batch, batch_size, d, d_index)

            x = self.fc_forward(x)

            return x