import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import GCNConv, Set2Set


class Set2SetNet(torch.nn.Module):
    def __init__(self, args, num_layers):
        super(Set2SetNet, self).__init__()
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.hidden = args.hidden_dim
        self.conv1 = GCNConv(self.num_features, self.hidden)
        self.convs = torch.nn.ModuleList()
        for i in range(num_layers - 1):
            self.convs.append(GCNConv(self.hidden, self.hidden))
        self.set2set = Set2Set(self.hidden, processing_steps=4)
        self.lin1 = Linear(2 * self.hidden, self.hidden)
        self.lin2 = Linear(self.hidden, self.hidden // 2)
        self.lin3 = Linear(self.hidden // 2, self.num_classes)

    def reset_parameters(self):
        self.conv1.reset_parameters()
        for conv in self.convs:
            conv.reset_parameters()
        self.set2set.reset_parameters()
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()

    def forward(self, data):
        x, edge_index, batch = data.x, data.edge_index, data.batch
        x = F.relu(self.conv1(x, edge_index))
        for conv in self.convs:
            x = F.relu(conv(x, edge_index))
        x = self.set2set(x, batch)
        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.lin3(x)
        return F.log_softmax(x, dim=-1)

    def __repr__(self):
        return self.__class__.__name__
