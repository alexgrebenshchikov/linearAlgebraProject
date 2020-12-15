import unittest
import numpy as np
from numpy import linalg as LA
import iterations_methods as IM
import QR_decomposition as QR
import search_eigvals as SE
import tridiagonalization_matrix as TM
import graphs as GR


def check_norm(m):
    LU = IM.LU_decomposition(m)
    L_inv = LA.inv(LU[0])
    U = LU[1]
    return LA.norm(L_inv @ U) < 1


def check_upper(m):
    for j in range(0, m.shape[0]):
        for i in range(j + 1, m.shape[1]):
            if abs(m[i][j]) > 1e-9:
                return False
    return True


def check_eigenvalues(A, res, self):
    for i in range(0, len(res[0])):
        self.assertTrue(np.allclose(A @ res[1][:, i: i + 1], res[0][i] * res[1][:, i: i + 1]))


class TestIterationsMethods(unittest.TestCase):
    def test_simple_iterations_method(self):
        n = 3
        a = np.random.rand(n, n)
        while not IM.check_Gerschgorin_rounds(a):
            a = np.random.rand(n, n)
        b = np.random.rand(n, 1)
        eps = 1e-6
        x = IM.iteration_method(a, b, eps)
        self.assertTrue(LA.norm(x - (a @ x) - b) < eps)

    def test_gauss_seidel_method(self):
        n = 3
        a = np.random.rand(n, n)
        while not check_norm(a) or IM.check_diag_zeros(a):
            a = np.random.rand(n, n)
        b = np.random.rand(n, 1)
        eps = 1e-6
        x = IM.Gauss_Seidel_method(a, b, eps)
        self.assertTrue(LA.norm((a @ x) - b) < eps)


class TestQRDecompositionMethods(unittest.TestCase):
    def test_qr_decomposition_1(self):
        a_1 = np.array([[1., 2., 4., 5.],
                        [3., 3., 2., 6.],
                        [4., 1., 3., 2.],
                        [5., 1., 6., 2.]])

        a_2 = np.array([[0., 2., 4., 5.],
                        [3., 0., 2., 6.],
                        [4., 1., 9., 2.],
                        [5., 1., 0., 2.]])
        res_1 = QR.QR_decomposition_1(a_1)
        res_2 = QR.QR_decomposition_1(a_2)
        self.assertTrue(check_upper(res_1[1]) and check_upper(res_2[1]))
        self.assertTrue(np.allclose(LA.inv(res_1[0]), res_1[0].transpose()) and
                        np.allclose(LA.inv(res_2[0]), res_2[0].transpose()))
        self.assertTrue(np.allclose(res_1[0] @ res_1[1], a_1) and np.allclose(res_2[0] @ res_2[1], a_2))

    def test_qr_decomposition_2(self):
        a = np.array([[1., 2., 4., 5.],
                      [3., 3., 2., 6.],
                      [4., 1., 3., 2.],
                      [5., 1., 6., 2.]])

        res = QR.QR_decomposition_2(a)
        self.assertTrue(check_upper(res[1]))
        self.assertTrue(np.allclose(LA.inv(res[0]), res[0].transpose()))
        self.assertTrue(np.allclose(res[0] @ res[1], a))


class TestSearchingEigenvaluesMethods(unittest.TestCase):
    def test_find_max_eigenvalue(self):
        n = 4
        a = np.random.rand(n, n)
        ans = SE.find_max_eigenvalue(a, SE.get_random_v(n), 1e-9)
        self.assertTrue(np.allclose(ans[0] * ans[1], a @ ans[1]))

    def test_find_eigenvalues(self):
        a = np.array([[1., 4., 5.],
                      [4., 2., 6.],
                      [5., 6., 3.]])
        res = SE.QR_algorithm(a)
        check_eigenvalues(a, res, self)


class TestTridiagonalization(unittest.TestCase):
    def test_tridiagonalization_matrix(self):
        a = np.array([[1., 4., 5., 7.],
                      [4., 2., 6., 8.],
                      [5., 6., 3., 9.],
                      [7., 8., 9., 7.]])
        q = TM.tridiagonalization(a)
        self.assertTrue(np.allclose(q[1].transpose() @ a @ q[1], q[0]))

    def test_QR_algorithm_for_tridiag_matrix(self):
        a = np.array([[1., 4., 0., 0.],
                      [4., 2., 6., 0.],
                      [0., 6., 3., 9.],
                      [0., 0., 9., 7.]])
        res = TM.QR_algorithm_for_tridiag_matrix(a)
        check_eigenvalues(a, res, self)

    def test_QR_algorithm_with_shifts(self):
        a = np.array([[1., 4., 0., 0., 0.],
                      [4., 2., 6., 0., 0.],
                      [0., 6., 3., 9., 0.],
                      [0., 0., 9., 7., 12.],
                      [0., 0., 0., 12., 11.]])
        res = TM.QR_algorithm_with_shifts(a)
        check_eigenvalues(a, res, self)


class TestGraphMethods(unittest.TestCase):
    def test_check_isomorphism_1(self):
        b = np.array([[0., 1., 1., 4., 1.],
                      [1., 0., 0., 2., 3.],
                      [1., 0., 2., 2., 2.],
                      [4., 2., 2., 0., 0.],
                      [1., 3., 2., 0., 0.]])

        c = np.array([[0., 1., 1., 1., 4.],
                      [1., 0., 0., 3., 2.],
                      [1., 0., 2., 2., 2.],
                      [1., 3., 2., 0., 0.],
                      [4., 2., 2., 0., 0.]])

        self.assertEqual(1, GR.check_isomorphism(b, c))

    def test_check_isomorphism_2(self):
        b = np.array([[4., 1., 3., 3., 2.],
                      [1., 0., 3., 1., 2.],
                      [3., 3., 4., 3., 3.],
                      [3., 1., 3., 4., 3.],
                      [2., 2., 3., 3., 2.]])

        c = np.array([[4., 1., 1., 3., 4.],
                      [1., 4., 2., 4., 2.],
                      [1., 2., 4., 0., 0.],
                      [3., 4., 0., 2., 1.],
                      [4., 2., 0., 1., 2.]])
        self.assertEqual(0, GR.check_isomorphism(b, c))

    def test_calculate_a_1(self):
        for n in range(2, 11):
            print("a_1({}) = {}\n".format(n, GR.calculate_a_1(n)))

    def test_calculate_a_2(self):
        primes = [2, 3, 5, 7, 11, 13, 17, 19, 23, 29]
        for p in primes:
            print("a_2({}) = {}\n".format(p, GR.calculate_a_2(p)))


if __name__ == '__main__':
    unittest.main()
