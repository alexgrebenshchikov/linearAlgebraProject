from math import sqrt

import numpy as np
from numpy import linalg as LA


def Givense_rotation(A, i, j, c, s):
    res_i = c * A[i] + s * A[j]
    res_j = c * A[j] - s * A[i]
    A[i] = res_i
    A[j] = res_j


def QR_decomposition_1(a):
    A = a.copy()
    n = A.shape[0]
    Q = np.eye(n)
    for j in range(0, n):
        flag = False
        first_not_null = -1
        for i in range(j, n):
            if A[i][j] != 0:
                if not flag:
                    flag = True
                    first_not_null = i
                else:
                    c = A[first_not_null][j] / (sqrt(A[first_not_null][j] ** 2 + A[i][j] ** 2))
                    s = A[i][j] / (sqrt(A[first_not_null][j] ** 2 + A[i][j] ** 2))
                    Givense_rotation(A, first_not_null, i, c, s)
                    Givense_rotation(Q, first_not_null, i, c, s)

        if flag and first_not_null != j:
            Givense_rotation(A, j, first_not_null, 0, 1)
            Givense_rotation(Q, j, first_not_null, 0, 1)

    return Q.transpose(), A


def Householder_reflection(A, v):
    return A - ((2 * v) @ (v.transpose() @ A)), np.eye(A.shape[0]) - 2 * (v @ v.transpose())


def add_zeros(u, j):
    for i in range(0, j):
        u[i] = 0


def QR_decomposition_2(A):
    n = A.shape[0]
    Q = np.eye(n)
    for j in range(0, n):
        u = A[:, j: j + 1].copy()
        add_zeros(u, j)
        if LA.norm(u) == 0:
            continue
        u = u / LA.norm(u)
        u[j] -= 1
        if LA.norm(u) == 0:
            u = np.zeros((n, 1))
            u[j] = 1
        else:
            u = u / LA.norm(u)
        A = Householder_reflection(A, u)[0]
        Q = Householder_reflection(Q, u)[0]
    return Q.transpose(), A


