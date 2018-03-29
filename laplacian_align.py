""" Manifold alignment using Laplacian Eigenmaps"""

import numpy as np
from scipy.linalg import eigh
from scipy.sparse.csgraph import laplacian
from scipy.spatial.distance import pdist, squareform
from sklearn.neighbors import kneighbors_graph


def laplacian_embed(X, k, sigma, d=2, norm=False):
    """ Laplacian eigenmap for one dataset

    Parameters
    ----------
    X     - High dimensional dataset
    k     - Graph will be of the k nearest neighbours
    sigma - Graph weighted by Gaussian kernel - 0 for unweighted edges
    d     - Embedding dimension
    norm  - If True, use normalised Laplacian matrix

    Returns - Low dimensional embedded coordinates
    """
    G = kneighbors_graph(X, k, include_self=False).toarray()
    G = (G + G.T) * 0.5 # symmetrise G
    if sigma == 0.:
        W = 1.
    else:
        W = squareform(pdist(X))
        W = np.exp(-W**2 / (2*sigma**2))
    G = G * W
    assert np.allclose(G, G.T), 'Assymmetric matrix!'
    L = laplacian(G, normed=norm)
    eig_vals, eig_vecs = eigh(L, eigvals=(1, d)) # this gives us 1, 2, ... , d evals
    return eig_vecs

def laplacian_manifold_align(Xs, Us, ks, sigmas, mu, d=2):
    """ Manifold alignment using Laplacian Eigenmaps

    Parameters
    ----------
    Xs     - List of X: High dimensional dataset, N_X x D_X array
    Us     - List of U: Inter-dataset similarity kernel. N_X x N_Y array
    ks     - Graphs are of k nearest neighbours - int to have constant, list to specify
    sigmas - Graphs weighted by Gaussian kernel or std dev sigma - float to have constant, list to specify
    mu     - Weighting of matching matrices compared to intra-dataset weights
    d      - Embedding dimension (default 2)

    Returns
    -------
    Z - Low dimensional embedded coordinates, (N_X + N_Y) x d array

    Notes
    -----
    Us is a simple list of lists with None on the diagonal such that
    U^{1,2} = Us[1][2]

    The U matrices must be symmetric, ie. Us[n][m] = Us[m][n].T for all n != m

    """
    n_layers = len(Xs)

    # extract shapes
    Ns, Ds = [], []
    for X in Xs:
        N, D = X.shape
        Ns.append(N)
        Ds.append(D)

    # we can use either a series of k, sigma values for the embedding
    # or use the same values for every slice
    if not hasattr(ks, '__iter__'):
        ks = [ks for s in range(n_layers)]
    if not hasattr(sigmas, '__iter__'):
        sigmas = [sigmas for s in range(n_layers)]

    # check symmetry of matrices
    for i in range(n_layers):
        for j in range(n_layers):
            if i != j:
                if not (Us[i][j] == Us[j][i].T).all():
                    raise ValueError('Assymmetric matching matrices %s and %s' % (i,j))

    # consruct graphs and graph Laplacians
    Ls = []
    for i in range(n_layers):
        G = kneighbors_graph(Xs[i], ks[i], include_self=False).toarray()
        G = (G + G.T) * 0.5 # symmetrise G
        W = squareform(pdist(Xs[i]))
        W = np.exp(-W**2 / (2*sigmas[i]**2))
        G = G * W
        L = laplacian(G, normed=False)
        L += np.diag(np.sum(L, axis=0)) * mu
        Ls.append(L)

    # construct big block matrix
    bmat_list = []
    for i in range(n_layers):
        bmat_list.append([])
        for j in range(n_layers):
            if i==j:
                bmat_list[-1].append(Ls[i])
            else:
                bmat_list[-1].append(Us[i][j] * -1 * mu)

    M = np.bmat(bmat_list)
    assert (M == M.T).all(), 'Block matrix is not symmetric'

    # extract primary eigenvectors as embedding coordinates
    eig_vals, eig_vecs = eigh(M, eigvals=(1, d)) # this gives us 1, 2, ... , d evals
    return eig_vecs
