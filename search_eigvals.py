import numpy as np
from numpy import linalg as LA
import QR_decomposition as QR


def get_random_v(n):
    rv = np.random.rand(n, 1)
    return rv / LA.norm(rv)


def find_max_eigenvalue(A, x, eps):
    lam = 0
    counter = 0
    while LA.norm((A @ x) - (lam * x)) >= eps:
        p = A @ x
        x = p / (LA.norm(p))
        lam = x.transpose() @ A @ x
        if counter > 1e6:
            return 0
        counter += 1
    return lam, x


def check_Gerschgorin_rounds(m, eps):
    for i in range(0, m.shape[0]):
        radius = 0
        for j in range(0, m.shape[1]):
            if i != j:
                radius += abs(m[i][j])
        if radius >= eps:
            return False
    return True


def check_upper_2(m):
    for j in range(0, m.shape[0]):
        for i in range(j + 1, m.shape[1]):
            if abs(m[i][j]) > 1e-9:
                return False
    return True


def QR_algorithm(A):
    n = A.shape[0]
    Q = np.eye(n)
    while not check_Gerschgorin_rounds(A, 1e-9):
        qr = QR.QR_decomposition_1(A)
        A = qr[1] @ qr[0]
        Q = Q @ qr[0]

    return [A[i][i] for i in range(n)], Q


