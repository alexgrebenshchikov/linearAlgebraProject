from math import sqrt

import QR_decomposition as QR
import numpy as np
from numpy import linalg as LA

import search_eigvals as SE


def double_Householder_reflection(A, v):
    t = A - ((2 * v) @ (v.transpose() @ A))
    return t - ((t @ (2 * v)) @ v.transpose())


def GV_matrix(n, i, j, c, s):
    res = np.eye(n)
    res[i][i] = c
    res[j][j] = c
    res[i][j] = s
    res[j][i] = -s
    return res


def right_Givense_rotation(A, i, j, c, s):
    res_i = np.zeros((A.shape[0], 1))
    res_j = np.zeros((A.shape[0], 1))
    for k in range(A.shape[0]):
        res_i[k] = c * A[k][i] + s * A[k][j]
        res_j[k] = c * A[k][j] - s * A[k][i]
    for k in range(A.shape[0]):
        A[k][i] = res_i[k]
        A[k][j] = res_j[k]


def QR_algorithm_step(A, Q):
    n = A.shape[0]
    Ac = A.copy()
    for j in range(0, n - 1):
        c = Ac[j][j] / (sqrt(Ac[j][j] ** 2 + Ac[j + 1][j] ** 2))
        s = Ac[j + 1][j] / (sqrt(Ac[j][j] ** 2 + Ac[j + 1][j] ** 2))
        QR.Givense_rotation(Ac, j, j + 1, c, s)
        QR.Givense_rotation(A, j, j + 1, c, s)
        right_Givense_rotation(A, j, j + 1, c, s)
        right_Givense_rotation(Q, j, j + 1, c, s)


def tridiagonalization(A):
    n = A.shape[0]
    Q = np.eye(n)
    for j in range(0, n - 1):
        u = A[:, j: j + 1].copy()
        QR.add_zeros(u, j + 1)
        if LA.norm(u) == 0:
            continue
        u = u / LA.norm(u)
        u[j + 1] -= 1
        if LA.norm(u) == 0:
            u = np.zeros((n, 1))
            u[j + 1] = 1
        else:
            u = u / LA.norm(u)
        A = double_Householder_reflection(A, u)
        Q = QR.Householder_reflection(Q, u)[0]
    return A, Q.transpose()


def QR_algorithm_for_tridiag_matrix(b):
    A = b.copy()
    n = A.shape[0]
    Q = np.eye(n)
    while not SE.check_Gerschgorin_rounds(A, 1e-9):
        QR_algorithm_step(A, Q)
    return [A[i][i] for i in range(n)], Q


def find_eigenvalue_22(A):
    D = (A[0][0] + A[1][1]) ** 2 - 4 * (A[0][0] * A[1][1] - A[0][1] * A[1][0])
    lam_1 = (A[0][0] + A[1][1] + sqrt(D)) / 2
    lam_2 = (A[0][0] + A[1][1] - sqrt(D)) / 2
    return lam_1 if abs(lam_1 - A[1][1]) < abs(lam_2 - A[1][1]) else lam_2


def check_last_line_and_column(A, eps):
    n = A.shape[0]
    radius1 = 0
    radius2 = 0
    for j in range(0, n - 1):
        radius1 += abs(A[n - 1][j])
    for i in range(0, n - 1):
        radius2 += abs(A[i][n - 1])
    return radius1 < eps and radius2 < eps


def QR_algorithm_with_shifts(b):
    A = b.copy()
    n = A.shape[0]
    Q = np.eye(n)
    for i in range(0, n - 1):
        while not check_last_line_and_column(A[0:n - i, 0:n - i], 1e-5):
            s = find_eigenvalue_22(A[n - 2 - i:n - i, n - 2 - i:n - i])
            A[0:n - i, 0:n - i] -= (s * np.eye(n - i))
            QR_algorithm_step(A[0:n - i, 0:n - i], Q)
            A[0:n - i, 0:n - i] += (s * np.eye(n - i))

    return [A[i][i] for i in range(n)], Q



