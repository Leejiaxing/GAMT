import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import get_laplacian, degree
from sklearn.model_selection import StratifiedKFold, KFold
import os
from torch_geometric.datasets import TUDataset
import torch_geometric.transforms as T


# function for pre-processing
@torch.no_grad()
def scipy_to_torch_sparse(A, device):
    A = sparse.coo_matrix(A)
    m = A.shape[0]
    n = A.shape[1]
    row = torch.tensor(A.row, dtype=torch.long, device=device)
    col = torch.tensor(A.col, dtype=torch.long, device=device)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data).to(device)

    return index, value, m, n


# function for pre-processing
def ChebyshevApprox(f, n):  # assuming f : [0, pi] -> R
    quad_points = 500
    c = np.zeros(n)
    a = np.pi / 2
    for k in range(1, n + 1):
        Integrand = lambda x: np.cos((k - 1) * x) * f(a * (np.cos(x) + 1))
        x = np.linspace(0, np.pi, quad_points)
        y = Integrand(x)
        c[k - 1] = 2 / np.pi * np.trapz(y, x)

    return c


# function for pre-processing
def get_operator(L, DFilters, n, s, J, Lev):
    r = len(DFilters)
    c = [None] * r
    for j in range(r):
        c[j] = ChebyshevApprox(DFilters[j], n)
    a = np.pi / 2  # consider the domain of masks as [0, pi]
    # Fast Tight Frame Decomposition (FTFD)
    FD1 = sparse.identity(L.shape[0])
    d = dict()
    for l in range(1, Lev + 1):
        for j in range(r):
            T0F = FD1
            T1F = ((s ** (-J + l - 1) / a) * L) @ T0F - T0F
            d[j, l - 1] = (1 / 2) * c[j][0] * T0F + c[j][1] * T1F
            for k in range(2, n):
                TkF = ((2 / a * s ** (-J + l - 1)) * L) @ T1F - 2 * T1F - T0F
                T0F = T1F
                T1F = TkF
                d[j, l - 1] += c[j][k] * TkF
        FD1 = d[0, l - 1]

    return d


# initialize dataset
def dataset_init(dataset, args):
    D1 = lambda x: np.cos(x / 2)
    D2 = lambda x: np.sin(x / 2)
    DFilters = [D1, D2]
    r = len(DFilters)
    args.r = r
    Lev = args.Lev
    s = args.s
    n = args.n
    dataset_temp = list()

    for i in range(len(dataset)):
        data_temp = Data(x=dataset[i].x, edge_index=dataset[i].edge_index, y=dataset[i].y)

        # get graph Laplacian
        num_nodes = data_temp.x.shape[0]
        L = get_laplacian(dataset[i].edge_index, num_nodes=num_nodes, normalization='sym')
        L = sparse.coo_matrix((L[1].numpy(), (L[0][0, :].numpy(), L[0][1, :].numpy())), shape=(num_nodes, num_nodes))

        # calculate lambda max
        lobpcg_init = np.random.rand(num_nodes, 1)
        lambda_max, _ = lobpcg(L, lobpcg_init)
        lambda_max = lambda_max[0]
        J = np.log(lambda_max / np.pi) / np.log(s) + Lev - 1  # dilation level to start the decomposition

        # get matrix operators
        d = get_operator(L, DFilters, n, s, J, Lev)
        for m in range(1, r):
            for q in range(Lev):
                if (m == 1) and (q == 0):
                    d_aggre = d[m, q]
                else:
                    d_aggre = sparse.vstack((d_aggre, d[m, q]))
        d_aggre = sparse.vstack((d[0, Lev - 1], d_aggre))
        data_temp.d = [d_aggre]

        # get d_index
        a = [i for i in range((r - 1) * Lev + 1)]
        data_temp.d_index = [torch.tensor([a[i // num_nodes] for i in range(len(a) * num_nodes)], device=args.device)]

        # append data1 into dataset1
        dataset_temp.append(data_temp)

    return dataset_temp


def K_Fold(folds, dataset, seed):
    skf = KFold(folds, shuffle=True, random_state=seed)
    test_indices = []
    for _, index in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(index)

    return test_indices


class NormalizedDegree(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, data):
        deg = degree(data.edge_index[0], dtype=torch.float)
        deg = (deg - self.mean) / self.std
        data.x = deg.view(-1, 1)
        return data


def get_dataset(name, sparse=True, cleaned=False):
    path = os.path.join('data', name)
    dataset = TUDataset(path, name, cleaned=cleaned)
    dataset.data.edge_attr = None

    if dataset.data.x is None:
        max_degree = 0
        degs = []
        for data in dataset:
            degs += [degree(data.edge_index[0], dtype=torch.long)]
            max_degree = max(max_degree, degs[-1].max().item())

        if max_degree < 1000:
            dataset.transform = T.OneHotDegree(max_degree)
        else:
            deg = torch.cat(degs, dim=0).to(torch.float)
            mean, std = deg.mean().item(), deg.std().item()
            dataset.transform = NormalizedDegree(mean, std)

    if not sparse:
        num_nodes = max_num_nodes = 0
        for data in dataset:
            num_nodes += data.num_nodes
            max_num_nodes = max(data.num_nodes, max_num_nodes)

        # Filter out a few really large graphs in order to apply DiffPool.
        if name == 'REDDIT-BINARY':
            num_nodes = min(int(num_nodes / len(dataset) * 1.5), max_num_nodes)
        else:
            num_nodes = min(int(num_nodes / len(dataset) * 5), max_num_nodes)

        indices = []
        for i, data in enumerate(dataset):
            if data.num_nodes <= num_nodes:
                indices.append(i)
        dataset = dataset[torch.tensor(indices)]

        if dataset.transform is None:
            dataset.transform = T.ToDense(num_nodes)
        else:
            dataset.transform = T.Compose(
                [dataset.transform, T.ToDense(num_nodes)])

    return dataset
