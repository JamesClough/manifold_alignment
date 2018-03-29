""" Useful functions that lots of other functions need"""

import matplotlib.pyplot as plt
import numpy as np
from scipy.spatial.distance import pdist, cdist, squareform
from numpy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.sparse.csgraph import laplacian
from sklearn.neighbors import kneighbors_graph

def normalise_manifold_weights(z, C):
    """ Given 1-array z, normalise it so min is 1. and max is C

    Notes
    -----
    Needs unit test"""
    z = np.array(z)
    z -= np.min(z) # set min to 0
    z *= (C-1) / np.max(z) # set range to C-1
    z += 1
    return z

def smooth_manifold_weights(z):
    """ Smooths manifold weights"""
    min_z = np.min(z)
    z += np.concatenate([[min_z], z[:-1]])
    z += np.concatenate([z[1:], [min_z]])
    return z

def tensor_to_vector(U):
    """ Symmetry: U[n,m,i,j] = U[m,n,j,i]
    And U[n,n,i,j] = 0

    So write V[z,i,j] where:
    z=0 -> n=1, m=0
    z=1 -> n=2, m=0
    z=2 -> n=2, m=1
    """
    N, _, I, J = U.shape

    # check 0s
    for n in range(N):
        if not (U[n,n] == 0).all():
            raise ValueError('Diagonal elements in %s not equal to 0' % n)

    # check symmetry
    for n in range(N):
        for m in range(N):
            if not (U[n,m] == U[m,n].T).all():
                raise ValueError('Tensor symmetry violated for %s-%s' % (n,m))

    z = 0
    len_z = (N * (N-1))/2
    V = np.zeros((len_z, I, J))
    for n in range(N):
        for m in range(N):
            if n > m:
                V[z] = U[n,m]
                z += 1
    return V

def vector_to_tensor(V):
    """ Inverse of tensor_to_vector function above """
    len_z, N, _ = V.shape
    num_slices = int(np.ceil(np.sqrt(2*len_z)))
    U = np.zeros((num_slices, num_slices, N, N))
    z = 0
    for n in range(num_slices):
        for m in range(num_slices):
            if n > m:
                U[n,m] = V[z]
                U[m,n] = V[z].T
                z += 1
    return U

def extract_wks_U(dist_matrices):
    """ We have a nice way of storing the WKS matching matrices that utilises
    the symmetry of the matchings. This saves half the space when writing those
    files but needs to be undone on the other side."""
    num_slices = len(dist_matrices)
    Us = [[] for i in range(num_slices)]
    for i in range(num_slices):
        for j in range(num_slices):
            if i==j:
                Us[i].append(None)
            elif i < j:
                Us[i].append((dist_matrices[i][j-i-1]))
            else:
                Us[i].append((dist_matrices[j][i-j-1].T))
    return Us

def matching_matrix_from_list(matching_col):
    matching_col = matching_col.astype(int)
    N = matching_col.shape[0]
    U_m = np.zeros([N,N])
    for ii in range(N):
        matching_row = range(N)
        U_m[matching_row[ii], matching_col[ii]] = 1.
    return U_m

def hungarian(U):
    """ Given matrix of weights, return binarised matrix of minimal elements

    Parameters
    ----------
    U - Weight matrix, numpy array

    Returns
    -------
    U_m - Binarised matrix of minimal matching

    Notes
    -----
    Scipy linear_sum_assignment is used to do the Hungarian algorithm
    There is a bug that is I think being fixed in the next version where
    a NaN in the input matrix U causes linear_sum_assignment to hang and needs
    the program to be killed.
    In the meantime we need to check for this manually and return something
    """
    N, M = U.shape
    if np.isnan(U).any():
        # invalid input
        return np.zeros((N,M))
    matching_row, matching_col = linear_sum_assignment(U)
    U_m = np.zeros([N,M])
    for ii in range(min(N,M)):
        U_m[matching_row[ii], matching_col[ii]] = 1.
    return U_m

def random_matching(N):
    import random
    """ Useful for benchmarking graph matching"""
    U_m = np.zeros([N,N])
    matching_row = range(N)
    matching_col = range(N)
    random.shuffle(matching_col)
    for ii in range(N):
        U_m[matching_row[ii], matching_col[ii]] = 1.
    return U_m

def calc_graph(X, k, sigma):
    """ Given data X construct graphs with k nearest neighbours and weighted
    by Gaussian kernel with std sigma

    Parameters
    ----------
    X     - array - TxQ array of T timepoints with Q features each
    k     - int   - number of nearest neighbours
    sigma - float - standard deviation of Gaussian kernel

    Returns
    -------
    TxT adjacency matrix of weighted graph

    Notes
    -----
    k=0 means complete graph
    sigma=0 means unweighted
    Can't do both"""
    assert isinstance(k, int), 'k must be an integer'
    T = X.shape[0]
    X = X.reshape(T, np.prod(X.shape[1:]))
    if k == 0 and sigma == 0:
        assert False, "Can't have k and sigma both equal to 0 - thats a complete unweighted graph"
    if k == 0:
        G = 1.
    else:
        G = kneighbors_graph(X, k, include_self=False)
        G = 0.5 * (G + G.T).toarray()
    if sigma == 0:
        W = 1.
    else:
        dist_G = squareform(pdist(X))
        W = np.exp(-(dist_G**2) / (2*sigma*sigma)) - np.identity(T)
    WG = G * W
    return WG


def calc_graphs(X, k, sigma):
    """ Given data X in some number of slices, construct graphs with
    k nearest neighbours and weighted by Gaussian kernel std sigma

    Set k=0 to have all neighbours, and sigma=0 to have unit weights
    But not both"""
    assert isinstance(k, int), 'k must be an integer'
    N = X.shape[0]
    Gs = []
    if k == 0 and sigma == 0:
        assert False, "Can't have k and sigma both equal to 0 - thats a complete unweighted graph"
    for i in range(N):
        # need to reshape this slice into a 2-array
        Gs.append(calc_graph(X[i], k, sigma))
    return Gs

def calc_graph_eigenvalues(Gs):
    """ Given list of graphs, Gs, return two lists of the eigenvalues
    and eigenvectos ot the graph laplacian for each"""
    G_eig_vals = []
    G_eig_vecs = []
    for G in Gs:
        eig_vals, eig_vecs = eigh(laplacian(G))
        G_eig_vals.append(eig_vals)
        G_eig_vecs.append(eig_vecs)
    G_eig_vals = np.array(G_eig_vals)
    G_eig_vecs = np.array(G_eig_vecs)
    return G_eig_vals, G_eig_vecs


def gaussian_similarity_kernel(F_x, F_y, sigma):
    """ Given two feature vectors form Gaussian similarity kernel"""
    Z = cdist(F_x, F_y)
    Z = Z ** 2
    Z /= (2 * sigma * sigma)
    Z *= -1
    Z = np.exp(Z)
    return Z

def gaussian_weighted_similarity(X, sigma):
    """ Given NxD coordinates X, return NxN similarity matrix

    Similarity between i and j is given by a Gaussian of their distance,
    with SD of sigma"""
    Z_compressed = pdist(X)            # Euclidean distances
    Z = squareform(Z_compressed)       # put it into matrix form
    Z = Z*Z
    Z /= 2 * sigma * sigma
    Z *= -1.
    Z = np.exp(Z)                      # now this is Gaussian weighted
    return Z

def reconstructed_image_distance(X_low, Y_low, Y_high, k):
    """ Given two aligned low-dimensional datasets X_low and Y_low each with N points
    we want to measure how well X can reconstruct Y.

    For each point X_low[i] in X_low, we find its k nearest neighbours in Y_low
    and measure their average distance in Y_high to Y_high[i], which is the actual
    corresponding point to X_low[i] in Y_high.
    We return the square mean
    """
    dists = cdist(X_low, Y_low)
    N, _ = X_low.shape
    results = []
    for i in range(N):
        Xi_nn = np.argsort(dists[i,:])
        results.append(np.mean(np.sum(Y_high[Xi_nn[:k]] - Y_high[i])**2, axis=1))
    return np.array(results)
