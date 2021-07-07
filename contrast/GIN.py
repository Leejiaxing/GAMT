import torch
import torch.nn.functional as F
from torch.nn import Linear, Sequential, ReLU, BatchNorm1d as BN
from torch_geometric.nn import GINConv, global_mean_pool


class GIN(torch.nn.Module):
    def __init__(self, args, num_layers):
        super(GIN, self).__init__()
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.hidden = args.hidden_dim
        self.conv1 = GINConv(
            Sequential(
                Linear(self.num_features, self.hidden),
                ReLU(),
                Linear(self.hidden, self.hidden),
                ReLU(),
                BN(self.hidden),
            ), train_eps=True)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(
                GINConv(
                    Sequential(
                        Linear(self.hidden, self.hidden),
                        ReLU(),
                        Linear(self.hidden, self.hidden),
                        ReLU(),
                        BN(self.hidden),
                    ), train_eps=True))
        self.lin1 = Linear(self.hidden, self.hidden)
        self.lin2 = Linear(self.hidden, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = self.conv1(x, edge_index)
        for conv in self.convs:
            x = conv(x, edge_index)
        x = global_mean_pool(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin2(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__