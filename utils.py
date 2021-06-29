import numpy as np
from scipy import sparse
from scipy.sparse.linalg import lobpcg
import torch
from torch_geometric.data import Data, DataLoader
from torch_geometric.utils import get_laplacian


# function for pre-processing
@torch.no_grad()
def scipy_to_torch_sparse(A, device='cuda:0'):
    A = sparse.coo_matrix(A)
    m = A.shape[0]
    n = A.shape[1]
    row = torch.tensor(A.row, dtype=torch.long, device='cuda:0')
    col = torch.tensor(A.col, dtype=torch.long, device='cuda:0')
    index = torch.stack((row, col), dim=0)
    value = torch.Tensor(A.data).cuda()

    # return torch.sparse_coo_tensor(index, value, A.shape).to(device)
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
def dataset_init(dataset, Lev, s, n, waveletType='Haar'):
    if waveletType == 'Haar':
        D1 = lambda x: np.cos(x / 2)
        D2 = lambda x: np.sin(x / 2)
        DFilters = [D1, D2]
    elif waveletType == 'Linear':
        D1 = lambda x: np.square(np.cos(x / 2))
        D2 = lambda x: np.sin(x) / np.sqrt(2)
        D3 = lambda x: np.square(np.sin(x / 2))
        DFilters = [D1, D2, D3]
    elif waveletType == 'Quadratic':  # not accurate so far
        D1 = lambda x: np.cos(x / 2) ** 3
        D2 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2)), np.cos(x / 2) ** 2)
        D3 = lambda x: np.multiply((np.sqrt(3) * np.sin(x / 2) ** 2), np.cos(x / 2))
        D4 = lambda x: np.sin(x / 2) ** 3
        DFilters = [D1, D2, D3, D4]
    else:
        raise Exception('Invalid FrameType')
    r = len(DFilters)

    dataset1 = list()
    for i in range(len(dataset)):
        data1 = Data(x=dataset[i].x, edge_index=dataset[i].edge_index, y=dataset[i].y)

        # get graph Laplacian
        num_nodes = data1.x.shape[0]
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
        data1.d = [d_aggre]
        # get d_index
        a = [i for i in range((r - 1) * Lev + 1)]
        data1.d_index = [torch.tensor([a[i // num_nodes] for i in range(len(a) * num_nodes)], device='cuda:0')]
        # append data1 into dataset1
        dataset1.append(data1)

    return dataset1, r
