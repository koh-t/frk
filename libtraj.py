# coding: utf-8

import numpy as np
from sklearn.metrics.pairwise import euclidean_distances

import _frechet
import _dtw
import _sdtw
import _ga
import pyfdk
import pygak
import pyfrk
from time import time
from hausdorff import hausdorff_distance


def distmat(x, y):
    D = euclidean_distances(x, y, squared=True)
    return D


eps = np.finfo(np.float).eps


class Hausdorff:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def compute(self):
        self.r = hausdorff_distance(self.x, self.y, distance="euclidean") ** 2


class DTW:
    def __init__(self, D):
        self.D = D

    def compute(self):
        [p, q] = self.D.shape
        D = self.D

        r = np.ones([p + 1, q + 1]) * np.inf
        r[0, :] = 0
        r[:, 0] = 0
        r[0, 0] = 0
        for i in range(1, p + 1):
            for j in range(1, q + 1):
                d = D[i - 1, j - 1]
                a = np.array([r[i - 1, j - 1], r[i - 1, j], r[i, j - 1]])
                r[i, j] = d + min(a)

        self.r = r

    def computec(self):
        [p, q] = self.D.shape
        r = np.zeros([p + 1, q + 1])
        self.r = _dtw._compute(self.D, r)


class Frechet:
    def __init__(self, D):
        self.D = D

    def compute(self):
        [p, q] = self.D.shape
        D = self.D

        C = -1 * np.ones([p, q])
        # r[:] = -1.0
        C[0, 0] = D[0, 0]
        for i in range(1, p):
            C[i, 0] = np.max([D[i, 0], C[i - 1, 0]])

        for j in range(1, q):
            C[0, j] = np.max([D[0, j], C[0, j - 1]])

        for i in range(1, p):
            for j in range(1, q):
                # cost = norm(x[i - 1, :], y[j - 1, :])
                a = [C[i - 1, j], C[i, j - 1], C[i - 1, j - 1]]
                # [r[i - 1, j],  # insertion
                # r[i, j - 1],  # deletion
                # r[i - 1, j - 1]])  # match
                C[i, j] = np.max([D[i, j], np.min(a)])

        self.r = C

    def computec(self):
        [p, q] = self.D.shape
        r = np.ones([p, q])
        self.r = _frechet._compute(self.D, r)


class SDTW:
    def __init__(self, D, _gamma=1.0):
        self.D = D
        self._gamma = _gamma

    def softmin(self, a, _gamma=1.0):
        a = -a / _gamma
        amax = a.max()
        tmp = np.exp(a - amax).sum()
        tmp = -_gamma * (np.log(tmp) + amax)
        return tmp

    def compute(self):
        [p, q] = self.D.shape
        D = self.D

        r = np.zeros([p + 1, q + 1])
        r[0, :] = np.inf
        r[:, 0] = np.inf
        r[0, 0] = 0.0

        for i in range(1, p + 1):
            for j in range(1, q + 1):
                a = np.array([r[i - 1, j - 1], r[i - 1, j], r[i, j - 1]])
                r[i, j] = D[i - 1, j - 1] + self.softmin(a, self._gamma)

        self.r = r

    def computec(self):
        [p, q] = self.D.shape
        r = np.zeros([p + 1, q + 1])
        self.r = _sdtw._compute(self.D, r, self._gamma)


class GA:
    def __init__(self, D, _gamma=1.0):
        self.D = D
        self._gamma = _gamma
        self.E = self.local(self.D, self._gamma)

    def gauss(self, D, _gamma=1.0):
        return np.exp(-D / _gamma)

    def local(self, D, _gamma=1.0):
        a = D / (2 * _gamma)
        b = np.log(2 - np.exp(-a) + eps)
        return np.exp(-(a + b))

    def compute(self):
        [p, q] = self.D.shape
        E = self.E

        r = np.ones([p + 1, q + 1]) * np.inf
        r[0, :] = 0.0
        r[:, 0] = 0.0
        r[0, 0] = 1.0

        for i in range(1, p + 1):
            for j in range(1, q + 1):
                a = [r[i - 1, j - 1], r[i - 1, j], r[i, j - 1]]
                r[i, j] = E[i - 1, j - 1] * np.sum(a)

        self.r = r

    def computec(self):
        [p, q] = self.D.shape
        r = np.zeros([p + 1, q + 1])
        self.r = _ga._compute(self.E, r)


class GAFrechet(object):
    def __init__(self, D, _gamma=1.0):
        self.D = D
        self._gamma = _gamma
        self.E = self.gauss(self.D, self._gamma)
        self.sort()

    def gauss(self, D, _gamma=1.0):
        # return np.exp(-D / _gamma)
        a = D / (2 * _gamma)
        b = np.log(2 - np.exp(-a) + eps)
        return np.exp(-(a + b))

    def sort(self):
        [p, q] = self.E.shape
        E = self.E
        row, col = np.indices((p, q))

        row = row.reshape([-1])
        col = col.reshape([-1])
        val = E[row, col]
        id_rank = val.argsort()
        row_sort = row[id_rank]
        col_sort = col[id_rank]
        r = np.zeros([p, q])
        for r_, id in enumerate(id_rank):
            row_ = row[id]
            col_ = col[id]
            r[row_, col_] = r_

        self.r = r
        self.row_sort = row_sort
        self.col_sort = col_sort

    def kernel(self):
        self.kernelval = (self.E[self.row_sort, self.col_sort] * self.c[0, 0, :]).sum() / np.sum(self.c[0, 0, :])

    def _compute(self):
        [p, q] = self.r.shape
        r = self.r

        c = np.zeros([p, q, p * q])
        r_ = int(r[-1, -1])
        c[-1, -1, r_] = 1

        # initialize last row
        i = p - 1
        for j_ in range(q - 1):
            j = q - j_ - 2
            rij = int(r[i, j])
            # smaller than rij
            for k_ in range(0, rij):
                a = [c[i, j + 1, k_]]
                c[i, j, k_] = np.sum(a)

            # equal to rij
            d = 0
            for k_ in range(rij, p * q):
                a = [c[i, j + 1, k_]]
                d += np.sum(a)
            c[i, j, rij] = d

        # initialize last col
        j = q - 1
        for i_ in range(p - 1):
            i = p - i_ - 2
            rij = int(r[i, j])
            # smaller than rij
            for k_ in range(0, rij):
                a = [c[i + 1, j, k_]]
                c[i, j, k_] = np.sum(a)

            # equal to rij
            d = 0
            for k_ in range(rij, p * q):
                a = [c[i + 1, j, k_]]
                d += np.sum(a)
            c[i, j, rij] = d

        # compute others
        for i_ in range(1, p):
            for j_ in range(1, q):
                i = p - i_ - 1
                j = q - j_ - 1
                rij = int(r[i, j])

                for k_ in range(rij):
                    a = [c[i, j + 1, k_], c[i + 1, j, k_], c[i + 1, j + 1, k_]]
                    c[i, j, k_] = np.sum(a)

                d = 0
                for k_ in range(rij, p * q):
                    a = [c[i, j + 1, k_], c[i + 1, j, k_], c[i + 1, j + 1, k_]]
                    d += np.sum(a)

                c[i, j, rij] = d

        self.c = c

    def compute(self):
        self._compute()
        self.kernel()

    def computec(self):
        [p, q] = self.r.shape
        self.c = np.zeros([p, q, p * q], dtype=np.int)
        self.c = _gafrechet._compute(self.r.astype(np.int), self.c)
        self.kernel()


class GAFrechetFlex(GAFrechet):
    def __init__(self, D, _gamma=1.0):
        super(GAFrechetFlex, self).__init__(D, _gamma)
        self.D = D
        self._gamma = _gamma
        self.E = self.gauss(self.D, self._gamma)
        self.sort()

    def kernel(self, _gamma):
        v = np.sum(self.c[0, 0, :])
        if v != 0.0:
            self.kernelval = (self.E[self.row_sort, self.col_sort] * (self.c[0, 0, :] / v)).sum()
        else:
            self.kernelval = (self.E[self.row_sort, self.col_sort] * (self.c[0, 0, :])).sum()

    def compute(self):
        self._compute()
        self.kernel(self._gamma)

    def computec(self):
        [p, q] = self.r.shape
        self.c = np.zeros([p, q, p * q], dtype=np.int)
        self.c = _gafrechet._compute(self.r.astype(np.int), self.c)
        self.kernel(self._gamma)


class GAFrechetBigFlex(GAFrechet):
    def __init__(self, D, _gamma=1.0):
        super(GAFrechetBigFlex, self).__init__(D, _gamma)

    def _kernel(self, _gamma):
        self.E = self.gauss(self.D, _gamma)
        e = self.E[self.row_sort, self.col_sort]
        c = self.c[0, 0, :]
        self.kernelval = _biggafrechet._kernel(e, c)

    def kernel(self, _gamma):
        self.E = self.gauss(self.D, _gamma)
        v = self.c[0, 0, :].sum()
        if v != 0.0:
            self.kernelval = (self.E[self.row_sort, self.col_sort] * (self.c[0, 0, :] / v)).sum()
        else:
            self.kernelval = (self.E[self.row_sort, self.col_sort] * (self.c[0, 0, :])).sum()

    def compute(self):
        self._compute()
        self.kernel(self._gamma)

    def computec(self):
        [p, q] = self.r.shape
        self.c = _biggafrechet._compute(self.r.astype(np.int))
        self.kernel(self._gamma)
        # self._kernel(self._gamma)


class mGAFrechet(GAFrechet):
    def __init__(self, D, _gamma=1.0):
        super(mGAFrechet, self).__init__(D, _gamma)

    def get_rankmatrix(self, y):
        x = y.copy()
        [p, q] = x.shape
        ind = np.unravel_index(np.argsort(x, axis=None), x.shape)
        x[ind] = np.arange(p * q)[::-1]
        return x

    def get_kernel(self):
        ind = np.array(self.ls)
        x = np.sort(self.E.flatten())
        v = x[ind[:, 0].astype(int)]
        k = (v * ind[:, 1]).sum()
        self.kernelval = k

    def merge(self, a, b, val):
        l = []  # Array.new
        i = 0  # index of a
        j = 0  # index of b

        # merge all elements that are smaller than val
        s = 0
        while i < len(a):
            if a[i][0] > val:
                break
            s += a[i][1]
            i += 1

        while j < len(b):
            if b[j][0] > val:
                break
            s += b[j][1]
            j += 1

        if s > 0:
            l.append([val, s])

        # merge the rest elements
        while i < len(a) and j < len(b):
            wa, ca = a[i]
            wb, cb = b[j]

            if wa == wb:
                l.append([wa, ca + cb])
                i += 1
                j += 1
            elif wa < wb:
                l.append([wa, ca])
                i += 1
            else:  # wa > wb
                l.append([wb, cb])
                j += 1

        while i < len(a):
            wa, ca = a[i]
            l.append([wa, ca])
            i += 1

        while j < len(b):
            wb, cb = b[j]
            l.append([wb, cb])
            j += 1

        return l

    def fdk_c(self):
        self.kernelval = _mgafrechet.fdk(self.E)

    def fdk(self):
        e = self.E
        [p, q] = self.E.shape
        # e = Array.new(len_a){|i| Array.new(len_b){|j| Math.exp( - (a[i] - b[j])**2) }}
        m = [[[]] * q for ii in range(p)]

        # compute m of L area of the DP table
        m[0][0] = [[e[0, 0], 1]]

        for i in range(1, p):
            m[i][0] = self.merge(m[i - 1][0], [], e[i, 0])

        for j in range(1, q):
            m[0][j] = self.merge(m[0][j - 1], [], e[0, j])

        # compute m of inside area of the DP table
        for i in range(1, p):
            for j in range(1, q):
                m[i][j] = self.merge(self.merge(m[i][j - 1], m[i - 1][j], e[i, j]), m[i - 1][j - 1], e[i, j])

        # evaluate kernel value \sum_{i, j} e_{i,j} c_{i,j}
        val = 0
        c = 0
        for i in range(p):
            for j in range(q):
                for m_ in m[i][j]:
                    val += m_[0] * m_[1]
                    c += m_[1]

        self.kernelval = val / c


class FDK:
    def __init__(self, D, beta=1.0, gamma=1.0, sample=100, diag_wgt=1.0, seed=0, ker4=''):
        self.beta = beta
        self.gamma = gamma
        self.sample = sample
        self.seed = seed
        self.diag_wgt = diag_wgt

        if ker4 == '':
            self.L = np.max(D.shape)
            self.ker4 = pyfdk.Kernel4(self.L)  # L is the maximum length of sequence
        else:
            self.ker4 = ker4

        self.D = D
        m, n = self.D.shape
        # self.E = self.gauss(self.D, self.gamma)
        self.E = self.local(self.D, self.gamma)

        # s = time()
        self.mat = pyfdk.MatrixF64(m, n)
        for i in range(m):
            for j in range(n):
                self.mat.set(i, j, self.E[i, j])
        # elap = time() - s
        # print("Elapsed: ", elap)

    def gauss(self, D, _gamma=1.0):
        return np.exp(-D / _gamma)

    def local(self, D, _gamma=1.0):
        a = D / (2 * _gamma)
        b = np.log(2 - np.exp(-a) + eps)
        return np.exp(-(a + b))

    def kernel(self):
        # Kernel Engine for k4
        # ker4 = pyfdk.Kernel4(self.L)  # L is the maximum length of sequence
        k4 = self.ker4.compute(mat=self.mat, nsamples=self.sample, beta=self.beta, diag_wgt=self.diag_wgt,
                               seed=self.seed)
        # k4 = pyfdk.compute_k4(mat=self.mat, nsamples=self.sample,
        #                       beta=self.beta, seed=self.seed)
        self.kernelval = k4


class logGAK:
    def __init__(self, x, y, sigma=1.0, triangular=0):
        self.x = x
        self.y = y
        self.nX = x.shape[0]
        self.nY = y.shape[0]
        self.dimvect = x.shape[1]
        self.sigma = sigma
        self.triangular = triangular

        self.x = x.T.flatten()
        self.y = y.T.flatten()
        self.seq1 = pygak.Sequence(self.nX * self.dimvect)  # Reserve nX * dimvect elements
        self.seq2 = pygak.Sequence(self.nY * self.dimvect)  # Reserve nY * dimvect elements

        # Set matrix values
        for i in range(len(self.seq1)):
            self.seq1.set(i, self.x[i])  # seq1[i] = random.random()
        for i in range(len(self.seq2)):
            self.seq2.set(i, self.y[i])  # seq2[i] = random.random()

    def kernel(self):
        self.kernelval = pygak.logGAK(self.seq1, self.seq2, self.nX, self.nY, self.dimvect, self.sigma, self.triangular)


class FRK:
    def __init__(self, D, beta=1.0, gamma=1.0, sample=100, diag_wgt=1.0, K=1, seed=0, ker4=''):
        self.beta = beta
        self.gamma = gamma
        self.sample = sample
        self.K = K
        self.seed = seed
        self.diag_wgt = diag_wgt

        if ker4 == '':
            self.L = np.max(D.shape)
            # self.ker4 = pyfdk.Kernel4(self.L)  # L is the maximum length of sequence
            self.ker4 = pyfrk.Kernel(self.L)  # L is the maximum length of sequence considered
        else:
            self.ker4 = ker4

        self.D = D
        m, n = self.D.shape
        # self.E = self.gauss(self.D, self.gamma)
        self.E = self.local(self.D, self.gamma)

        # s = time()
        self.mat = pyfrk.MatrixF64(m, n)
        for i in range(m):
            for j in range(n):
                self.mat.set(i, j, self.E[i, j])
        # elap = time() - s
        # print("Elapsed: ", elap)

    def gauss(self, D, _gamma=1.0):
        return np.exp(-D / _gamma)

    def local(self, D, _gamma=1.0):
        a = D / (2 * _gamma)
        b = np.log(2 - np.exp(-a) + eps)
        return np.exp(-(a + b))

    def kernel(self):
        self.tf = pyfrk.TopkFrechet(self.mat)
        for k in range(self.K):
            self.tf.next()  # if there is no next point, will return False

        k4 = self.ker4.compute_with_topk(emat=self.mat, tf=self.tf, nsamples=self.sample,
                                         beta=self.beta, diag_wgt=self.diag_wgt, seed=self.seed)
        # Kernel Engine for k4
        # k4 = self.ker4.compute(mat=self.mat, nsamples=self.sample, beta=self.beta, diag_wgt=self.diag_wgt,
        #                        seed=self.seed)
        # k4 = pyfdk.compute_k4(mat=self.mat, nsamples=self.sample,
        #                       beta=self.beta, seed=self.seed)
        self.kernelval = k4
