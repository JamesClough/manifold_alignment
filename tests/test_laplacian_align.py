import numpy as np
from nose.tools import assert_almost_equals, assert_true, assert_raises
from manifold_alignment.laplacian_align import laplacian_manifold_align, laplacian_embed
import manifold_alignment.utils as utils

from sklearn import datasets

class TestLaplacianEmbed(object):
    def test_shape(self):
        N = 100
        X, X_m = datasets.make_s_curve(N, random_state=0)
        for d in [2,3,4]:
            L = laplacian_embed(X, k=5, sigma=0.1, d=d)
            assert_true(L.shape==(N, d))

class TestLaplacianAlign(object):
    def test_shape(self):
        """ Check L has the correct shape"""
        N = 100
        X, X_m = datasets.make_s_curve(N, random_state=0)
        Y, Y_m = datasets.make_s_curve(N, random_state=1)
        U_1 = utils.gaussian_similarity_kernel(X_m.reshape(N,1), Y_m.reshape(N,1), 1.0)
        U_2 = utils.gaussian_similarity_kernel(Y_m.reshape(N,1), X_m.reshape(N,1), 1.0)
        for d in [2,3,4]:
            L = laplacian_manifold_align([X,Y], [[None, U_1], [U_2, None]], 5, 0.1, 0.1, d=d)
            assert_true(L.shape==(N*2, d))

    def test_assymetric_U(self):
        """ Should raise error when U is assymetric"""
        N = 100
        X, X_m = datasets.make_s_curve(N, random_state=0)
        Y, Y_m = datasets.make_s_curve(N, random_state=1)
        # make some assymetric matching matrices
        U_1 = utils.gaussian_similarity_kernel(X_m.reshape(N,1), Y_m.reshape(N,1), 1.0)
        U_2 = utils.gaussian_similarity_kernel(Y_m.reshape(N,1), X_m.reshape(N,1), 0.5)
        assert_raises(ValueError, laplacian_manifold_align, [X,Y], [[None, U_1], [U_2, None]], 5, 0.1, 0.1)

    def test_diff_size_graphs(self):
        N_X = 100
        N_Y = 200
        X, X_m = datasets.make_s_curve(N_X, random_state=0)
        Y, Y_m = datasets.make_s_curve(N_Y, random_state=1)
        U_1 = utils.gaussian_similarity_kernel(X_m.reshape(N_X,1), Y_m.reshape(N_Y,1), 1.0)
        U_2 = utils.gaussian_similarity_kernel(Y_m.reshape(N_Y,1), X_m.reshape(N_X,1), 1.0)
        for d in [2,3,4]:
            L = laplacian_manifold_align([X,Y], [[None, U_1], [U_2, None]], 5, 0.1, 0.1, d=d)
            assert_true(L.shape==(N_X + N_Y, d))
