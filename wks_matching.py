""" Wave kernel signature graph matching"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from tqdm import tqdm

MACHINE_EPSILON = np.finfo(np.double).eps

def wks_vec(ts, i, eig_vals, phis, sigma, drop_first=True, norm=True):
    """ Calculate wave kernel signature (in t, the energy scale)
    for node i in the graph.

    Parameters
    ----------
    ts         - array of t values
    i          - index of node in graph
    eig_vals   - eigenvalues of Laplacian of the graph
    phis       - eigenvectors of the Laplacian of the graph
    sigma      - smoothness of the wave kernel signature
    drop_first - if True, then do not use the first eigenvalue (since it is 0)
    norm       - if True, then normalise the signatures

    Returns
    -------
    Array of same length as ts, of the wave kernel signature for this node i
    """
    eig_vals[eig_vals<0] = MACHINE_EPSILON
    if drop_first:
        phis = phis[:,1:]
        eig_vals = eig_vals[1:]
    phi_i = phis[i,:]
    # Gaussian of difference between t and log of eigenvalues
    exp_vec = np.exp(-(np.subtract.outer(np.log(eig_vals), ts))**2 / (2*sigma**2))
    terms = exp_vec.T * phi_i**2
    if norm:
        exp_vec_norm = np.sum(exp_vec, axis=0)
        exp_vec_norm[exp_vec_norm <= 0] = MACHINE_EPSILON # avoid divide by 0 later
        return np.sum(terms, axis=1) / (exp_vec_norm)
    else:
        return np.sum(terms, axis=1)

def wks_full_vec(ts, G_eig_vals, G_eig_vecs, sigma, drop_first=True, norm=True):
    """ Calculate all wks vectors for a set of graphs of same size

    Parameters
    ----------
    ts         - array of t values
    G_eig_vals - eigenvalues of Laplacian of the graphs
    G_eig_vecs - eigenvectors of the Laplacian of the graphs
    sigma      - smoothness of the wave kernel signature
    drop_first - if True, then do not use the first eigenvalue (since it is 0)
    norm       - if True, then normalise the signatures

    Returns
    -------
    N x T x num_t array
    where [n,i,:] is the wks signature for graph n and vertex i
    """
    N, T = G_eig_vecs.shape[:2]
    # number of eigenvals / eigenvecs not necessarily equal to N!
    num_t = len(ts)
    wks_full_vecs = np.zeros((N, T, num_t))
    for n in range(N):
        for i in range(T):
            wks_full_vecs[n,i,:] = wks_vec(ts, i, G_eig_vals[n], G_eig_vecs[n], sigma, drop_first=drop_first, norm=norm)
    return wks_full_vecs

def wks_t_range(G_eig_vals, sigma, num_t, C=2, method='extremal', drop_first=True):
    """ Return array of t values for wks calculation

    Parameters
    ----------
    G_eig_vals - NxT array of graph eigenvalues
    sigma      - wks smoothness parameter
    num_t      - number of t values to generate
    C          - number of sigma_wks either side of min and max eigenvalues to use
    method     - how to find min and max eigenvalues from the whole set
    drop_first - if True, remove first eigenvalue from each set

    Returns
    -------
    ts - num_t length array

    Notes
    -----
    Only uses graph eigenvalues that are above 0.
    """
    if drop_first:
        if len(G_eig_vals.shape) > 1:
            G_eig_vals = G_eig_vals[:,1:]
        else:
            G_eig_vals = G_eig_vals[1:]
    if method == 'extremal':
        G_eig_vals_nz = G_eig_vals[G_eig_vals > MACHINE_EPSILON]
        t_min = np.min(np.log(np.array(G_eig_vals_nz)))
        t_max = np.max(np.log(np.array(G_eig_vals_nz)))
    elif method == 'mean':
        min_eig_vals = []
        max_eig_vals = []
        for n in range(G_eig_vals.shape[0]):
            G_eig_vals_n = G_eig_vals[n]
            G_eig_vals_n_nz = G_eig_vals_n[G_eig_vals_n > MACHINE_EPSILON]
            min_eig_vals.append(G_eig_vals_n_nz[0])
            max_eig_vals.append(G_eig_vals_n_nz[-1])
        t_min = np.mean(np.log(np.array(min_eig_vals)))
        t_max = np.mean(np.log(np.array(max_eig_vals)))
    t_min -= (C * sigma)
    t_max += (C * sigma)
    ts = np.linspace(start=t_min, stop=t_max, num=num_t)
    return ts

def wks_matching_tensor(G_eig_vals, G_eig_vecs, sigma, num_t, drop_first=True, norm=True):
    """ Wave kernel signature matching between many graphs of the same size.

    Parameters
    ----------
    G_eig_vals - numpy array of graphs eigenvalues
    G_eig_vecs - numpy array of graphs eigenvectors
    sigma      - smoothness of the wave kernel signature
    num_t      - number of steps in numerical integration
    drop_first - if True, then do not use the first eigenvalue (since it is 0)
    norm       - if True, then normalise the signatures
    """
    N, T = G_eig_vecs.shape[:2]

    if drop_first:
        G_eig_vals = G_eig_vals[:,1:]
        G_eig_vecs = G_eig_vecs[:,:,1:]

    ts = wks_t_range(G_eig_vals, sigma, num_t, drop_first=False)
    wks_full_vecs = wks_full_vec(ts, G_eig_vals, G_eig_vecs, sigma, drop_first=False, norm=True)
    U = np.zeros((N, N, T, T))
    for n in range(N):
        for m in range(N):
            if n > m:
                for i in range(T):
                    wks_diff = np.sum(np.abs(wks_full_vecs[n,i,:] - wks_full_vecs[m,:,:]), axis=1)
                    wks_sum = np.sum(np.abs(wks_full_vecs[n,i,:] + wks_full_vecs[m,:,:]), axis=1)
                    wks_ratio = wks_diff / wks_sum
                    U[n,m,i,:] = wks_ratio
                    U[m,n,:,i] = wks_ratio
    return U
