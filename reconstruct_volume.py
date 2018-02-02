""" Reconstruct volume from low-dimensional aligned manifold

Notes
-----

X - first index determines the slice
  - second index determines the time point
  - other indices describe the image

L - first index determines the slice
  - second index determines the time point
  - other indices describe coords in low-dimensional space

n - determines which slice to use as the base of the reconstructed volume
    images in other slices are chosen to have a similar motion state as the
    images in this slice.
    n should be chosen to be a slice which shows lots of motion and can be easily
    matched to other slices.

"""

import numpy as np

def reconstruct(X, L, n):
    """ Reconstruct volume from slice n. Returns in X-space.

    Parameters
    ----------
    X - high-dimensional dataset
    L - low-dimensional aligned manifold
    n - slice to use for reconstruction

    Returns
    -------
    Y - reconstructed volume
    """
    N, T = X.shape[:2]
    if N!=L.shape[0] or T!=L.shape[1]:
        raise AssertionError("""Shape mismatch between high-dimensional dataset
                             and low-dimensional manifold.\n
                             High-dimensional data: %sx%s
                             Low-dimensional manifold: %sx%s
                             """ % (N, T, L.shape[0], L.shape[1]))

    Y = np.zeros_like(X)
    for t in range(T):
        # reconstruct volume at time t
        for m in range(N):
            L_dists = np.sum((L[m,:] - L[n,t])**2, axis=1)
            nn_indices = np.argsort(L_dists)
            Y[m,t] = X[m,nn_indices[0]]
    return Y
