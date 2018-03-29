import numpy as np
from nose.tools import assert_almost_equals, assert_equals, assert_true, assert_raises
import manifold_alignment.utils as utils

class TestHungarian(object):
    def test_shape(self):
        np.random.seed(0)
        X = np.random.randn(10, 10)
        H = utils.hungarian(X)
        assert_true(X.shape == H.shape)

    def test_assymetric(self):
        np.random.seed(0)
        X = np.random.randn(12, 10)
        H = utils.hungarian(X)
        assert_true(X.shape == H.shape)
        assert_equals(np.sum(H), 10)

        X = np.random.randn(13, 15)
        H = utils.hungarian(X)
        assert_true(X.shape == H.shape)
        assert_equals(np.sum(H), 13)

    def test_example(self):
        X = np.array([[0, 2, 3, 4],
                     [4, 1, 2, 2],
                     [3, 3, 3, 2],
                     [5, 2, 1, 5]])
        H = utils.hungarian(X)
        X = np.array([[1, 0, 0, 0],
                     [0, 1, 0, 0],
                     [0, 0, 0, 1],
                     [0, 0, 1, 0]])
        assert_true((X==H).all())

    def test_nan(self):
        X = np.array([[0, 2, 3, 4],
                     [4, 1, 2, np.nan],
                     [3, 3, 3, 2]])
        H = utils.hungarian(X)
        assert_true((H==np.zeros((3,4))).all())

class TestCalcGraphs(object):
    def test_shape(self):
        N = 10
        M = 35
        D = 6
        np.random.seed(0)
        X = np.random.randn(N, M, D)
        Gs = utils.calc_graphs(X, k=5, sigma=0.1)
        assert_equals(len(Gs), N)
        assert_equals(Gs[0].shape[0], M)
        assert_equals(Gs[0].shape[1], M)

    def test_multidim(self):
        N = 10
        M = 35
        D1 = 6
        D2 = 5
        np.random.seed(0)
        X = np.random.randn(N, M, D1, D2)
        Gs = utils.calc_graphs(X, k=5, sigma=0.1)
        assert_equals(len(Gs), N)
        assert_equals(Gs[0].shape[0], M)
        assert_equals(Gs[0].shape[1], M)

class TestCalcGraphEigenvalues(object):
    def test_shape(self):
        N = 20
        M = 45
        D = 4
        np.random.seed(0)
        X = np.random.randn(N, M, D)
        Gs = utils.calc_graphs(X, k=6, sigma=0.2)
        G_eig_vals, G_eig_vecs = utils.calc_graph_eigenvalues(Gs)
        assert_equals(G_eig_vals.shape, (N, M))
        assert_equals(G_eig_vecs.shape, (N, M, M))
        for i in range(N):
            assert_almost_equals(G_eig_vals[i,0], 0)


class TestTensorVectorConversion(object):
    def test_conversion(self):
        N, S = 10, 20
        np.random.seed(0)
        U = np.zeros((N, N, S, S))
        for i in range(N):
            for j in range(N):
                if i > j:
                    M = np.random.randn(S, S)
                    U[i,j] = M
                    U[j,i] = M.T
        V = utils.tensor_to_vector(U)
        U_ = utils.vector_to_tensor(V)
        assert_true((U_ == U).all())

    def test_shape_V(self):
        N, S = 20, 30
        np.random.seed(0)
        U = np.zeros((N, N, S, S))
        for i in range(N):
            for j in range(N):
                if i > j:
                    M = np.random.randn(S, S)
                    U[i,j] = M
                    U[j,i] = M.T
        V = utils.tensor_to_vector(U)
        assert_equals(len(V.shape), 3)
        assert_equals(V.shape[1], S)
        assert_equals(V.shape[2], S)
        assert_equals(V.shape[0], ((N*(N-1))/2))

    def test_shape_U(self):
        N, S = 25, 35
        np.random.seed(0)
        V = np.random.randn(((N*(N-1))/2), S, S)
        U = utils.vector_to_tensor(V)
        assert_equals(len(U.shape), 4)
        assert_equals(U.shape[0], N)
        assert_equals(U.shape[1], N)
        assert_equals(U.shape[2], S)
        assert_equals(U.shape[3], S)

    def test_symmetry_error(self):
        N, S = 12, 15
        np.random.seed(0)
        U = np.zeros((N, N, S, S))
        for i in range(N):
            for j in range(N):
                if i != j:
                    M = np.random.randn(S, S)
                    U[i,j] = M
        assert_raises(ValueError, utils.tensor_to_vector, U)

    def test_zero_error(self):
        N, S = 12, 15
        np.random.seed(0)
        U = np.zeros((N, N, S, S))
        for i in range(N):
            for j in range(N):
                if i >= j:
                    M = np.random.randn(S, S)
                    U[i,j] = M
                    U[j,i] = M.T
        assert_raises(ValueError, utils.tensor_to_vector, U)
