import numpy as np
import scipy.spatial.distance as distance
import torch

def pairwise(x, y, method='euclidean', M=None, nan_val=99999):
    D = np.array(())
    if method == 'dotp':
        f = lambda u, v: np.dot(u,v) # Dot product for each pair of vectors
        D = - distance.cdist(x, y, metric=f) # Get the dot product back to a dissimilarity value (just take the negative of the similarity...)
    elif method == 'euclidean' or method == 'sqeuclidean':
        D = pytorch_pairwise_euclidean(x, y, method)
    else:
        D = distance.cdist(x, y, metric=method, VI=M)

    # Fix NaN's
    D[np.isnan(D)] = nan_val
    return D

def pytorch_pairwise_euclidean(x, y=None, method='euclidean'):
    '''
    Input: x is a Nxd matrix
          y is an optional Mxd matirx
    Output: dist is a NxM matrix where dist[i,j] is the square norm between x[i,:] and y[j,:]
           if y is not given then use 'y=x'.
    i.e.     dist[i,j] = ||x[i,:]-y[j,:]||^2 (if method=='sqeuclidean')
            dist[i,j] = ||x[i,:]-y[j,:]|| (if method=='euclidean')
    '''
    # TODO =>  UNSTABLE!! WRONG DISSIMILARITY VALUES IF INPUT IS NOT L2 NORMALIZED!!!
    x = torch.from_numpy(x.copy())
    x_norm = x.norm(p=2, dim=1, keepdim=True)
    if y is None:
        y = x
    y = torch.from_numpy(y.copy())
    y_norm = y.norm(p=2, dim=1, keepdim=True).transpose(0,1)

    # Dist
    dist = x_norm + y_norm - 2.0 * torch.matmul(x, y.transpose(0, 1))

    # Ensure distance is in [0, inf]
    dist = torch.clamp(dist, 0.0, np.inf)

    # Euclidean or squared euclidean?
    if method == 'euclidean':
        dist = torch.sqrt(dist)

    return dist.numpy()


def pytorch_pairwise_euclidean2(x, y=None, method='euclidean'):
    if y is None:
        y = x

    x = torch.from_numpy(x.copy())
    y = torch.from_numpy(y.copy())

    dist = torch.zeros(x.size(0), y.size(0))
    for i, row in enumerate(x.split(1)):
        r_v = row.expand_as(y)
        dist[i] = torch.sum((r_v - y) ** 2, 1).view(1, -1)

    #dists = Parallel(n_jobs=20)(delayed(_square_distance)(row, y) for row in x.split(1))
    #dist = torch.cat(dists)

    # Euclidean or squared euclidean?
    if method == 'euclidean':
        dist = torch.sqrt(dist)

    return dist


def _square_distance(x, y):
    r_v = x.expand_as(y)
    return torch.sum((r_v - y) ** 2, 1).view(1, -1)