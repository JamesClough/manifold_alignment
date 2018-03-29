import numpy as np
from numpy.linalg import eigh
from scipy.sparse.csgraph import laplacian

from nose.tools import assert_equals, assert_almost_equals, assert_true
import manifold_alignment.wks_matching as wks
import manifold_alignment.utils as utils

class TestWksTrange(object):
    def test_t_range(self):
        G_eig_vals = np.array([np.arange(0.1, 3.0, 0.1), np.arange(0.2, 3.1, 0.1)])
        sigma = 0.1
        num_t = 500
        C = 5
        ts = wks.wks_t_range(G_eig_vals, sigma, num_t, C, drop_first=False)
        assert_equals(len(ts), num_t)
        assert_almost_equals(ts[0], np.min(np.log(np.array(G_eig_vals))) - (C * sigma))
        assert_almost_equals(ts[-1], np.max(np.log(np.array(G_eig_vals))) + (C * sigma))

        ts = wks.wks_t_range(G_eig_vals, sigma, num_t, C, drop_first=True)
        assert_equals(len(ts), num_t)
        assert_almost_equals(ts[0], np.min(np.log(np.array(G_eig_vals[:,1:]))) - (C * sigma))
        assert_almost_equals(ts[-1], np.max(np.log(np.array(G_eig_vals))) + (C * sigma))

    def test_neg_eval(self):
        G_eig_vals = np.arange(-1., 3.0, 0.1)
        sigma = 0.1
        num_t = 500
        C = 5
        ts = wks.wks_t_range(G_eig_vals, sigma, num_t, C, drop_first=False)
        assert_almost_equals(ts[0], (np.log(0.1) - (C * sigma)))

class TestWksVec(object):
    def test_shape(self):
        A = np.array([[0,0,1,0,0,0,],
                     [0,0,0,1,0,0,],
                     [1,0,0,1,1,0,],
                     [0,1,1,0,1,1,],
                     [0,0,1,1,0,1,],
                     [0,0,0,1,1,0,]])
        eig_vals, eig_vecs = eigh(laplacian(A))
        ts = wks.wks_t_range(eig_vals, 1., 500)
        S  = wks.wks_vec(ts, 0, eig_vals, eig_vecs, 1.)
        assert_equals(len(S), 500)

    def test_shape_dropped_evals(self):
        A = np.array([[0,0,1,0,0,0,],
                     [0,0,0,1,0,0,],
                     [1,0,0,1,1,0,],
                     [0,1,1,0,1,1,],
                     [0,0,1,1,0,1,],
                     [0,0,0,1,1,0,]])
        eig_vals, eig_vecs = eigh(laplacian(A))
        eig_vals = eig_vals[1:]
        eig_vecs = eig_vecs[:,1:]
        ts = wks.wks_t_range(eig_vals, 1., 500)
        S  = wks.wks_vec(ts, 0, eig_vals, eig_vecs, 1.)
        assert_equals(len(S), 500)

class TestWksFullVec(object):
    def test_shape(self):
        A = np.array([[0,0,1,0,0,0,],
                     [0,0,0,1,0,0,],
                     [1,0,0,1,1,0,],
                     [0,1,1,0,1,1,],
                     [0,0,1,1,0,1,],
                     [0,0,0,1,1,0,]])

        B = np.array([[0,1,1,0,0,1,],
                     [1,0,0,1,0,0,],
                     [1,0,0,1,1,0,],
                     [0,1,1,0,1,1,],
                     [0,0,1,1,0,1,],
                     [1,0,0,1,1,0,]])

        Gs = [A,B]
        G_eig_vals, G_eig_vecs = utils.calc_graph_eigenvalues(Gs)
        G_eig_vals = G_eig_vals[:,1:]
        G_eig_vecs = G_eig_vecs[:,:,1:]
        ts = wks.wks_t_range(G_eig_vals, 1., 500)
        wks_full_vecs = wks.wks_full_vec(ts, G_eig_vals, G_eig_vecs, 1.0)
        assert_equals((2,6,500), wks_full_vecs.shape)

class TestWaveKernelMatching(object):
    def test_isomorphism(self):
        """ Test whether two isomorphic graphs can be perfectly matched.
        Note - the graph used must have no automorphisms - ie. there
        can only be one correct answer and so its harder to test."""

        A = np.array([[0,0,1,0,0,0,],
                     [0,0,0,1,0,0,],
                     [1,0,0,1,1,0,],
                     [0,1,1,0,1,1,],
                     [0,0,1,1,0,1,],
                     [0,0,0,1,1,0,]])

        B = A.copy()
        Gs = [A,B]
        G_eig_vals, G_eig_vecs = utils.calc_graph_eigenvalues(Gs)

        T = wks.wks_matching_tensor(G_eig_vals, G_eig_vecs, sigma=1., num_t=1000)
        for i in range(6):
            for j in range(6):
                if i==j:
                    assert_almost_equals(T[0, 1, i, j], 0.)
                else:
                    assert_true(T[0, 1, i,j] > 0)
