import scipy.sparse.linalg
import scipy.spatial.distance
from sklearn.cluster import SpectralClustering
import scipy
import scipy.sparse as sp
import numpy as np
import copy
import torch
from scipy.sparse import coo_matrix
import torch
import torch.nn.functional as F
from torch_geometric.datasets import TUDataset
from torch_geometric.data import DataLoader, Data
from torch_geometric.nn import GCNConv
from torch.utils.data import random_split
import os
import os.path as osp
import argparse
import warnings

def tree_idx(tree, j1, J1, J2):
    """
    tree_idx(treeG,j1,J1,J2) finds the index in level J2 of the j1-th node in
     level J1 for the tree treeG, where J1<J2<numel(treeG).

     INPUTS
     j1,J1  -  index of the j1-th node in level J1
     J2     -  higer level

     OUTPUT
     j2     -  index of the node in level J2
     """
    j = j1
    for k in np.arange(J1 + 1, J2 + 1, 1):
        j = tree[k]['IDX'][j]

    j2 = j
    return j2


def tree_idx2(treeG, k1, J1, J2):
    """
    finds all indices at level J2 of the k1th node of level J1 for J2<J1
    here the graph at higher level is coarser in the chain

    INPUTS:
      (k1,J1) - k1th node of level J1 (graph G_{J1})
      J2     - level J2 of the chain
    OUTPUTS:
      y      - indices of all nodes whose parent at level J1 is k1
    """
    g = treeG[J1]['clusters'][k1]
    if (J1 > J2 + 1):
        for j in np.arange(J2 + 1, J1)[::-1]:
            g1 = []
            for i in np.arange(0, len(g), 1):
                g1 = np.array(np.append(g1, treeG[j]['clusters'][g[i]]), dtype=int)
            g = g1
    y = g
    return y


def SC_tree(adjacency_matrix, levels, ratio=0.5):
    """
    Coarsen a graph multiple times using the SpectralClustering algorithm.
    INPUT
        W: symmetric sparse weight (adjacency) matrix
        levels: the number of coarsened graphs
    OUTPUT
        treeG with three objects:
        - IDX (presenting parents) at each level
        - clusters at each level
        - coarsened adjaency matrix at each level
    """
    parents = []
    adj_list = []
    W = adjacency_matrix
    N_start, N_start = W.shape
    adj_list.append(sp.csr_matrix(W))
    for k in range(levels - 1):
        # levels==0 stands for the original graph
        # PAIR THE VERTICES AND CONSTRUCT THE ROOT VECTOR
        idx_row, idx_col, val = scipy.sparse.find(W)
        perm = np.argsort(idx_row)
        rr = idx_row[perm]
        cc = idx_col[perm]
        vv = val[perm]
        """
        Using SpectralClustering at each level
        """
        if W.shape[0] <= 2 or k == levels - 2:
            sc = SpectralClustering(n_clusters=1, affinity='precomputed', n_init=10)
        else:
            sc = SpectralClustering(n_clusters=int(W.shape[0] * ratio) + 1, affinity='precomputed', n_init=10)
        if W.shape[0] == 1:
            cluster_id = np.array([0])
        else:
            cluster_id = sc.fit(W).labels_
        parents.append(cluster_id)
        # COMPUTE THE EDGES WEIGHTS FOR THE NEW GRAPH
        nrr = cluster_id[rr]
        ncc = cluster_id[cc]
        nvv = vv
        Nnew = cluster_id.max() + 1
        # CSR is more appropriate: row,val pairs appear multiple times
        W = sp.csr_matrix((nvv, (nrr, ncc)), shape=(Nnew, Nnew))
        adj_list.append(W)  # saving the coarsened adj

    """
    Constructing the tree 
    """
    treeG = list(np.arange(0, levels, 1))
    # Obtain the clusters and IDX, saved in treeG
    for i in range(levels):
        if i == 0:  # special case for the base level, corresponding to the original graph
            parents_ini = np.arange(0, N_start, 1);
            idx_sort = np.argsort(parents_ini)
            sorted_records_array = parents_ini[idx_sort]
            vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                               return_index=True)
            cluster_1 = np.split(idx_sort, idx_start[1:])
            treeG[i] = {'IDX': np.arange(0, N_start, 1), 'clusters': cluster_1, 'adj': adj_list[0]}
        else:  # the second level to the top level
            idx_sort = np.argsort(parents[i - 1])
            sorted_records_array = parents[i - 1][idx_sort]
            vals, idx_start, count = np.unique(sorted_records_array, return_counts=True,
                                               return_index=True)
            cluster_temp = np.split(idx_sort, idx_start[1:])
            treeG[i] = {'IDX': parents[i - 1], 'clusters': cluster_temp, 'adj': adj_list[i]}
    return treeG


def HaarGOB(treeG):
    """
    HaarGOB generates Haar Global Orthonormal Basis for a chain or tree treeG
#     %
    INPUTS:
#     %  treeG  - chain or tree for a graph
#     % OUTPUTS:
#     %  treeG1 - the updated chain treeG which is added for each level a Haar
#     %           orthonormal basis and for the bottom level the basis is the
#     %           Haar Global Orthonormal Basis
    """
    # number of level of the chain (or tree)
    Ntr = len(treeG)
    # reorder chain (optional)
    # reordering each level so that in each level the nodes are in the
    # descent order of degrees
    # compute u_l^c for level J_0 (top level)
    clusterJ0 = treeG[Ntr - 1]['clusters']
    N0 = len(clusterJ0)
    # generate indicator function on G^c
    chic = np.identity(N0)
    uc = [None] * N0
    uc[0] = 1 / np.sqrt(N0, dtype=np.float64) * np.ones(N0)
    for l in np.arange(1, N0):
        uc[l] = np.sqrt((N0 - l) / (N0 - l + 1)) * (chic[l - 1, :] - 1 / (N0 - l) * np.sum(chic[l:, :], axis=0))
    #    u = copy.deepcopy(uc)
    treeG[Ntr - 1]['u'] = uc
    # compute the next level orthonormal basis ulk and stored into u
    for j_tr in np.arange(0, Ntr - 1)[::-1]:
        N1 = len(treeG[j_tr]['clusters'])
        u = [None] * N1
        i = N0
        for l in range(N0):
            clusterl = treeG[j_tr + 1]['clusters'][l]
            kl = len(clusterl)
            # for k==1
            ucl = uc[l]
            ul1 = np.zeros(N1)
            for j in range(N0):
                idxj = treeG[j_tr + 1]['clusters'][j]
                ul1[idxj] = ucl[j] / np.sqrt(len(idxj))
            u[l] = ul1
            if kl > 1:
                chil = np.zeros((kl, N1))
                for k in range(kl):
                    idxl = treeG[j_tr + 1]['clusters'][l]
                    chil[k, idxl[k]] = 1;

                for k in np.arange(1, kl):
                    i = i + 1
                    ulk = np.sqrt((kl - k) / (kl - k + 1)) * (
                                chil[k - 1, :] - 1 / (kl - k) * np.sum(chil[k:, :], axis=0))
                    u[i - 1] = ulk
        treeG[j_tr]['u'] = u
        # update uc and N0
        #        uc = copy.deepcopy(u)
        uc = u
        N0 = N1
    return treeG


def adj2edge(adj):
    """
    adj2edge computes edge_index and edge_weights for adjacency matrix adj
    INPUTS:
        adj         - adjacency matrix of a graph, in coo_matrix format
    OUTPUTS:
        edge_index  - list for indices of edge ends
        edge_weight - non-zero values/elements in adj
    """
    adj = adj.tocoo().astype(np.float64)
    row = adj.row
    col = adj.col
    values = adj.data
    edge_weights = torch.Tensor(values)
    edge_index = torch.LongTensor([list(row), list(col)])
    return edge_index, edge_weights


def edge2adj(edge_index, edge_weight, num_nodes):
    """
    edge2adj computes adjacency matrix by edge_index and edge_weights
    INPUTS:
        edge_index  - list for indices of edge ends
        edge_weight - non-zero values/elements in adj
    OUTPUTS:
        adj         - adjacency matrix
    """
    adj = torch.sparse.FloatTensor(edge_index, edge_weight, torch.Size([num_nodes, num_nodes]))
    return adj


def Uext(edge_index, num_nodes, levels, ratio=0.5, edge_weight=None):
    """
    Uext computes the extended Haar basis for each layer of the tree.
        It generates a tree with L layers and then Haar basis for each layer.
    Inputs:
        edge_index      - edge_index for adjacency matrix
        edge_weight     - edge_weight for adjacency matrix
        num_nodes       - number of nodes of the graph
        L               - number of layers of tree
    Outputs:
        U_edge_index    - list of edge index for extended Haar basis for layer 1,...,L-1
        U_edge_weight   - list of edge weights for extended Haar basis for layer 1,...,L-1
    """
    # generate tree
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1),),
                                 device=edge_index.device)
    A = edge2adj(edge_index, edge_weight, num_nodes).to_dense()
    A = A.cpu().numpy()
    tree = SC_tree(A, levels, ratio)
    # generate Haar basis for each layer of tree
    tree_Haar = HaarGOB(tree)
    Nv = np.zeros([len(tree_Haar)], dtype='int64')  # number of nodes of chain
    U = list(np.arange(0, len(tree_Haar), 1))
    edge_index_list = list(np.arange(0, len(tree_Haar) - 1, 1))
    #     U[0] = np.array(tree_Haar[0]['u']) # Global Haar basis
    U[0] = [];
    num_edges_tree = np.zeros(levels, dtype=np.int)
    num_nodes_tree = np.zeros(levels, dtype=np.int)
    for j in np.arange(0, len(tree_Haar) - 1, 1):
        # adjaency matrix for layer j
        edge_index_list[j], _ = adj2edge(coo_matrix(tree[j + 1]['adj']))
        # compute number of edges for each layer
        num_edges_tree[j] = len(edge_index_list[j][1])
        # compute number of nodes for each layer
        num_nodes_tree[j] = len(tree[j]['clusters'])
        u = tree_Haar[j]['u']
        N = len(u)
        N1 = len(tree_Haar[j + 1]['u'])
        Nv[j + 1] = N1
        HaarBases = np.zeros((N, N1), dtype=np.float64)
        for k in np.arange(0, N1, 1):
            HaarBases[:, k] = u[k]
        #                HaarBases[K[l],k] = u[k][l]/np.sqrt(len(K[l]))
        U[j + 1] = HaarBases  #
    num_edges_tree[-1] = 1
    num_nodes_tree[-1] = 1
    return U, num_nodes_tree, num_edges_tree, edge_index_list


def Uext_batch(x, edge_index, batch, batch_size, num_node, num_edge, levels):
    """
    Compute extended Haar basis for batch
    INPUT:
        x                 - input data for batch, size = [Num_nodes Num_features]
        edge_index        - edge index of input graph for batch, size = [2 Num_edges]
        batch             - index of nodes of each sample in the batch
        batch_size        - number of samples in the batch
        num_node          - number of nodes of each sample in the batch
        num_edge          - number of edges of each sample in the batch
        levels            - number of layers of tree
    OUTPUT:
        U                 - list of list of adjacency matrix for all layers and all samples in batch
        edge_index_list   - list of edge_index_list
        num_nodes_tree    - list of number of nodes for all layers of tree
        num_edges_tree    - list of number of edges for all layers of tree
    """
    U = list()
    edge_index_list = list()
    num_nodes_tree = list()
    num_edges_tree = list()
    for i in range(batch_size):
        # number of nodes for i-th sample in the batch
        # compute the order index for edges of i-th sample in the batch
        idx_i1 = sum(num_edge[0:i])
        idx_i = sum(num_edge[0:(i + 1)])
        if i == 0:
            edge_index_i = edge_index[:, idx_i1:idx_i]
        else:
            edge_index_i = edge_index[:, idx_i1:idx_i] - sum(num_node[0:i])
        u_i, num_nodes_tree_i, num_edges_tree_i, edge_index_list_i = Uext(edge_index_i, num_node[i], levels)
        U.append(u_i)
        edge_index_list.append(edge_index_list_i)
        num_nodes_tree.append(num_nodes_tree_i)
        num_edges_tree.append(num_edges_tree_i)
    return U, edge_index_list, num_nodes_tree, num_edges_tree


def coo2st(coo):
    """
    transform matrix in sparse coo_matrix to sparse tensor of torch
    INPUT
     coo - matrix in sparse coo_matrix format
    OUTPUT
     coo matrix in torch.sparse.tensor
    """
    values = coo.data
    indices = np.vstack((coo.row, coo.col))
    i = torch.LongTensor(indices)
    v = torch.FloatTensor(values)
    shape = coo.shape
    return torch.sparse.FloatTensor(i, v, torch.Size(shape))


def HaarPool(x, edge_index, U, edge_index_list, num_nodes_tree, batch, batch_size, L):
    """
    HaarPooling
    INPUT:
        x                 - input data for batch, size = [Num_nodes Num_features]
        edge_index        - edge index of input graph for batch, size = [2 Num_edges]
        U                 - Extended Haar basis, U[i][L] is the extended Haar basis
                            for Lth layer of the tree and ith sample
        edge_index_list   - edge
        num_nodes_tree    - list of number of nodes of tree
        batch             - index of nodes of each sample in the batch
        batch_size        - number of samples in the batch
        L                 - number of pooling layer
    OUTPUT:
        x_1               - pooled x in batch
        edge_index_1      - pooled edge_index in batch
        batch_1           - new batch after pooling
    """
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda:0' if use_cuda else 'cpu')
    for i in range(batch_size):
        bi = (batch == i)  # compute samples of batch which equate to i
        ui = coo2st(coo_matrix(U[i][L])).transpose(0, 1)  # turn U to sparse tensor and transpose
        ui = ui.to(device)
        xi = torch.sparse.mm(ui, x[bi, :])
        edge_index_i = edge_index_list[i][L - 1]  # extract the edge_index of Lth layer of tree for ith sample of batch
        s_num_node_i = 0
        for j in range(i):
            s_num_node_i = s_num_node_i + num_nodes_tree[j][L]
        n_i = num_nodes_tree[i][L]
        batch_i = batch[bi][0:n_i]
        if i == 0:
            x_1 = xi
            edge_index_1 = edge_index_i
            batch_1 = batch_i
        if i > 0:
            x_1 = torch.cat([x_1, xi], dim=0)
            edge_index_i = edge_index_i + s_num_node_i
            edge_index_1 = torch.cat([edge_index_1, edge_index_i], dim=1)
            batch_1 = torch.cat([batch_1, batch_i], dim=0)
        edge_index_1.to(device)
    return x_1, edge_index_1, batch_1




warnings.filterwarnings("ignore")  # ignore all warnings

dataname = 'PROTEINS'
path = osp.join(os.path.abspath(''), 'data', dataname)  # ENZYMES/ DD/
dataset = TUDataset(path, name=dataname)
dataset = dataset.shuffle()

num_features = dataset.num_features
num_classes = dataset.num_classes

dataset1 = list()
for i in range(len(dataset)):
    data1 = Data(x=dataset[i].x, edge_index= \
        dataset[i].edge_index, y=dataset[i].y)
    data1.num_node = dataset[i].num_nodes
    data1.num_edge = dataset[i].edge_index.size(1)
    dataset1.append(data1)

dataset = dataset1



# %%
##Define Model Class
class Net(torch.nn.Module):
    def __init__(self, args):
        super(Net, self).__init__()
        self.nhid = args.hidden_dim
        self.num_layers = 3
        self.num_classes = args.num_classes
        self.num_features = args.num_features
        self.device = args.device
        self.conv1 = GCNConv(self.num_features, self.nhid)
        self.conv2 = GCNConv(self.nhid, self.nhid)
        self.conv3 = GCNConv(self.nhid, self.nhid)
        self.conv4 = GCNConv(self.nhid, self.nhid)

        self.lin1 = torch.nn.Linear(self.nhid, self.nhid)
        self.lin2 = torch.nn.Linear(self.nhid, self.nhid // 2)
        self.lin3 = torch.nn.Linear(self.nhid // 2, self.num_classes)

    def forward(self, data):
        x, edge_index, batch, num_node, num_edge = data.x, data.edge_index, data.batch, data.num_node, data.num_edge
        batch_size = int(batch.max() + 1)
        U, edge_index_list, num_nodes_tree, num_edges_tree = \
            Uext_batch(x, edge_index, batch, batch_size, num_node, num_edge, self.num_layers + 1)

        x = F.relu(self.conv1(x, edge_index.to(self.device)))
        x = F.relu(self.conv2(x, edge_index.to(self.device)))
        x, edge_index, batch = \
            HaarPool(x, edge_index, U, edge_index_list, num_nodes_tree, batch, batch_size, 1)

        x = F.relu(self.conv2(x, edge_index.to(self.device)))
        x = F.relu(self.conv2(x, edge_index.to(self.device)))
        x, edge_index, batch = \
            HaarPool(x, edge_index, U, edge_index_list, num_nodes_tree, batch, batch_size, 2)

        x = F.relu(self.conv3(x, edge_index.to(self.device)))
        x = F.relu(self.conv3(x, edge_index.to(self.device)))
        x, edge_index, batch = \
            HaarPool(x, edge_index, U, edge_index_list, num_nodes_tree, batch, batch_size, 3)

        x = F.relu(self.conv4(x, edge_index.to(self.device)))

        x = F.relu(self.lin1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.lin2(x))
        x = F.log_softmax(self.lin3(x), dim=-1)

        return x



