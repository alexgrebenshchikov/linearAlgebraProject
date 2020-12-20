import numpy as np
import search_eigvals as SE
import tridiagonalization_matrix as TM


def check_degrees_different(g_1, g_2):
    n_1 = g_1.shape[0]
    n_2 = g_2.shape[0]
    if n_1 != n_2:
        return True
    ds_1 = {}
    ds_2 = {}
    for i in range(0, n_1):
        d_1 = 0
        d_2 = 0
        for j in range(0, n_1):
            d_1 += g_1[i][j]
            d_2 += g_2[i][j]
        if d_1 not in ds_1:
            ds_1[d_1] = 0
        if d_2 not in ds_2:
            ds_2[d_2] = 0
        ds_1[d_1] += 1
        ds_2[d_2] += 1
    return ds_1 != ds_2


def check_isomorphism(g_1, g_2):
    if check_degrees_different(g_1, g_2):
        return 0
    tr_1 = TM.tridiagonalization(g_1)[0]
    tr_2 = TM.tridiagonalization(g_2)[0]
    sp_1 = TM.QR_algorithm_with_shifts(tr_1)[0]
    sp_2 = TM.QR_algorithm_with_shifts(tr_2)[0]
    sp_1.sort()
    sp_2.sort()
    if not np.allclose(sp_1, sp_2):
        return 0
    return 1


def add_edge(g, i, j):
    g[i][j] += 1.


def check_regularity(g):
    n = g.shape[0]
    cur_d = -1
    for i in range(n):
        s = 0
        for j in range(n):
            s += g[i][j]
        if cur_d != -1 and s != cur_d:
            return False
        cur_d = s
    return True


def build_graph_1(g, n):
    for x in range(n):
        for y in range(n):
            u = x * n + y
            add_edge(g, u, ((x + 2 * y) % n) * n + y)
            add_edge(g, u, ((x - 2 * y + 2 * n) % n) * n + y)
            add_edge(g, u, ((x + (2 * y + 1)) % n) * n + y)
            add_edge(g, u, ((x - (2 * y + 1) + 2 * n) % n) * n + y)
            add_edge(g, u, x * n + ((y + 2 * x) % n))
            add_edge(g, u, x * n + ((y - 2 * x + 2 * n) % n))
            add_edge(g, u, x * n + ((y + (2 * x + 1)) % n))
            add_edge(g, u, x * n + ((y - (2 * x + 1) + 2 * n) % n))


def build_graph_2(g, p):
    for i in range(p + 1):
        rev = euclid_ext(i, p)[1] % p if i != 0 and i != p else p - i
        nxt = (i + 1) % p if i != p else p
        prv = (i + p - 1) % p if i != p else p
        add_edge(g, i, rev)
        add_edge(g, i, nxt)
        add_edge(g, i, prv)


def calculate_a(g, d):
    q = TM.tridiagonalization(g)
    eigs = TM.QR_algorithm_with_shifts(q[0])[0]
    eigs.remove(max([abs(i) for i in eigs]))
    return max([abs(i) for i in eigs]) / d


def calculate_a_1(n):
    g = np.zeros((n * n, n * n))
    build_graph_1(g, n)

    if not check_regularity(g):
        return

    d = sum(g[0][i] for i in range(n * n))
    return calculate_a(g, d)


def calculate_a_2(p):
    g = np.zeros((p + 1, p + 1))
    build_graph_2(g, p)

    if not check_regularity(g):
        return

    d = sum(g[0][i] for i in range(p + 1))
    return calculate_a(g, d)


def euclid_ext(a, b):
    if b == 0:
        return a, 1, 0
    else:
        d, x, y = euclid_ext(b, a % b)
        return d, y, x - y * (a // b)
