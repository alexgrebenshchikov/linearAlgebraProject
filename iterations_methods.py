import numpy as np
from numpy import linalg as LA


def check_Gerschgorin_rounds(m):
    for i in range(0, m.shape[0]):
        center_x = 0
        radius = 0
        for j in range(0, m.shape[1]):
            if i == j:
                center_x = m[i][j]
            else:
                radius += abs(m[i][j])

        if center_x + radius >= 1:
            return False
    return True


def LU_decomposition(A):
    L = np.zeros((A.shape[0], A.shape[1]))
    U = np.zeros((A.shape[0], A.shape[1]))
    for i in range(0, A.shape[0]):
        for j in range(0, A.shape[1]):
            if j > i:
                U[i][j] = A[i][j]
            else:
                L[i][j] = A[i][j]
    return L, U


def check_diag_zeros(A):
    for i in range(0, A.shape[0]):
        if A[i][i] == 0:
            return True
    return False


def iteration_method(A, b, eps):
    is_good_eig = check_Gerschgorin_rounds(A)
    x = np.random.rand(A.shape[0], 1)
    fail_counter = 0
    while LA.norm(x - (A @ x) - b) >= eps:
        prev_x = x
        x = (A @ x) + b
        fail_counter = fail_counter + 1 if LA.norm(x) >= LA.norm(prev_x) + 1 else 0

        if fail_counter == 20 and not is_good_eig:
            return 0
    return x


def Gauss_Seidel_method(A, b, eps):
    is_zeros_on_diag = check_diag_zeros(A)
    if is_zeros_on_diag:
        return 0
    x = np.random.rand(A.shape[0], 1)
    LU = LU_decomposition(A)
    L_inv = LA.inv(LU[0])
    U = LU[1]
    fail_counter = 0
    while LA.norm((A @ x) - b) >= eps:
        prev_x = x
        x = L_inv @ (-(U @ x) + b)
        fail_counter = fail_counter + 1 if LA.norm(x) >= LA.norm(prev_x) + 1 else 0

        if fail_counter == 20:
            return 0
    return x
