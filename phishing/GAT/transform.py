import torch
import torch.nn.functional as F


def zero_resize(A: torch.tensor, shape):
    if A.dim() != len(shape):
        print('dim error')
        raise Exception
    for i in range(A.dim()):
        if A.shape[i] > shape[i]:
            # print('reshape,A.shape:',A.shape,' after:',shape)
            if len(A.shape) == 2 and len(shape) == 2:
                A = A[:shape[0], :shape[1]]
            else:
                raise Exception
    for i in range(A.dim()):
        s = list(A.shape)
        s[i] = shape[i] - A.shape[i]
        if s[i] == 0:
            continue
        A = torch.cat([A, torch.zeros(s)], dim=i)
    return A


def npstrArraytoFloatArray2D(array, base=16):
    if array.size <= 0:
        print("size of array <=0")
        return None
    if array.ndim != 2:
        print("ndim error")
        print("array.ndim:", array.ndim)
        return None
    # array=array.astype('<U'+len(int(int(array.dtype.str[2:])*'f',16).__str__()).__str__())
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            if array[i, j] != '':
                array[i, j] = int(array[i, j][0:min(len(array[i,j]),6)], base=base)
            elif array[i, j] == '':
                array[i, j] = 0
    return array





def col_norm(X: torch.tensor):
    X = F.normalize(X, p=1, dim=1)
    return X


def normalize(X: torch.tensor):
    if X.ndim == 2:
        col_sum = X.sum(0)
        D = torch.diag(col_sum)
        D = torch.pow(D, -0.5)
        D[torch.isinf(D)] = 0
        X = D @ X @ D
    elif X.ndim == 3:
        for i in range(X.shape[0]):
            col_sum = X[i].sum(0)
            D = torch.diag(col_sum)
            D = torch.pow(D, -0.5)
            D[torch.isinf(D)] = 0
            X[i] = D @ X[i] @ D
    else:
        raise ValueError
    return X


def dropedge(E: torch.tensor, rate=0.5):
    """
    E须为方阵
    """
    if E.shape[E.ndim - 1] == E.shape[E.ndim - 2]:
        t = torch.randn(E.shape)
        t = t @ t.transpose(t.ndim - 1, t.ndim - 2)  # t服从卡方分布
        t = t.ge(rate ** 2).float()
        print(t.shape)
        print(E.shape)
        E = E * t
        return E
    else:
        raise ValueError  # 传入参数E不是方阵，或遇到其他问题


def GCN_Atransform(A: torch.tensor):
    return A + torch.eye(A.shape[0], A.shape[1])
