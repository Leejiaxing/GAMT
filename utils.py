import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import get_laplacian, degree
from sklearn.model_selection import StratifiedKFold
from math import floor


# function for pre-processing
@torch.no_grad()
def scipy_to_torch_sparse(A, device='cuda:0'):
    A = sparse.coo_matrix(A)
    m = A.shape[0]
    n = A.shape[1]
    row = torch.tensor(A.row, dtype=torch.long, device=device)
    col = torch.tensor(A.col, dtype=torch.long, device=device)
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data).to(device)

    return index, value, m, n
    # A = sparse.coo_matrix(A)
    # row = torch.tensor(A.row, device=device)
    # col = torch.tensor(A.col, device=device)
    # index = torch.stack((row, col), dim=0)
    # value = torch.Tensor(A.data).to(device)
    #
    # return torch.sparse_coo_tensor(index, value, A.shape).to(device)


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


# Compute feature
def com_feature(dataset):
    dataset_temp = list()
    for i in range(len(dataset)):
        x = degree(dataset[i].edge_index[0], dataset[i].num_nodes).view(-1, 1)
        data_i = Data(x=x, edge_index=dataset[i].edge_index, y=dataset[i].y)
        dataset_temp.append(data_i)

    return dataset_temp


# initialize dataset
def dataset_init(dataset, args):
    D1 = lambda x: np.cos(x / 2)
    D2 = lambda x: np.sin(x / 2)
    DFilters = [D1, D2]
    r = len(DFilters)
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

    return dataset_temp, r


def K_Fold(folds, dataset):
    skf = StratifiedKFold(folds, shuffle=True, random_state=0)

    test_indices, train_indices = [], []
    for _, idx in skf.split(torch.zeros(len(dataset)), dataset.data.y):
        test_indices.append(idx)

    # val_indices = [test_indices[i - 1] for i in range(folds)]
    # np.ones()
    # for i in range(folds):
    #     train_mask = np.ones(len(dataset), dtype=bool)
    #     train_mask[test_indices[i]] = 0
    #     train_mask[val_indices[i]] = 0
    #     train_indices.append(train_mask.nonzero().view(-1))

    return test_indices


def K_fold(k, len):
    split = []
    counter = 0
    block = len / k
    while counter < len - 0.5:
        c = counter + block
        split.append(list(range(floor(counter + 0.5), floor(c + 0.5))))
        counter = c
    all_index = list(range(len))
    np.random.shuffle(all_index)
    return [np.take(all_index, i, axis=0).tolist() for i in split]