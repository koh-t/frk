# coding: utf-8
import os
import argparse
import numpy as np
import pickle as pkl
import pandas as pd
from tqdm import tqdm
from multiprocessing import Pool
from multiprocessing import sharedctypes
from sklearn.metrics.pairwise import euclidean_distances

import pyfrk
eps = np.finfo(np.float).eps

os.environ["OMP_NUM_THREADS"] = "1"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "1"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "1"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "1"  # export NUMEXPR_NUM_THREADS=6


def distmat(x, y):
    D = euclidean_distances(x, y, squared=True)
    return D


class FRK:
    def __init__(self, D, beta=1.0, gamma=1.0, sample=100, diag_wgt=1.0, K=1, seed=0, ker4='', method='frk1'):
        self.beta = beta
        self.gamma = gamma
        self.sample = sample
        self.K = K
        self.seed = seed
        self.diag_wgt = diag_wgt
        self.method = method

        if ker4 == '':
            self.L = np.max(D.shape)
            # L is the maximum length of sequence considered
            self.ker4 = pyfrk.Kernel(self.L)
        else:
            self.ker4 = ker4

        self.D = D
        m, n = self.D.shape
        self.E = self.gauss(self.D, self.gamma)
        # self.E = self.local(self.D, self.gamma)

        self.mat = pyfrk.MatrixF64(m, n)
        for i in range(m):
            for j in range(n):
                self.mat.set(i, j, self.E[i, j])

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

        if self.method == 'frk1':
            k4 = self.ker4.compute(emat=self.mat, nsamples=self.sample,
                                   beta=self.beta, diag_wgt=self.diag_wgt, seed=self.seed)
        elif self.method == 'frk2':
            k4 = self.ker4.compute_with_topk(emat=self.mat, tf=self.tf, nsamples=self.sample,
                                             beta=self.beta, diag_wgt=self.diag_wgt, seed=self.seed)
        self.kernelval = k4


def run(i, j, gamma, beta=1.0, diag_wgt=1.0, sample=100, K=1, method='frk1'):
    x = df.traj[i]
    y = df.traj[j]
    D = distmat(x, y)
    obj = FRK(D, beta=beta, gamma=gamma, diag_wgt=diag_wgt,
              sample=sample, K=K, method=method)
    obj.kernel()
    tmp = np.ctypeslib.as_array(shared_array)
    tmp[i, j] = obj.kernelval


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='computing grammatrix with kdtw')
    # parser.add_argument('--data', help='dataname',  default='170309-Omizunagidori-Preprocessed', type=str)
    # parser.add_argument('--data', help='dataname',  default='Geolife\ Trajectories\ 1.3', type=str)
    parser.add_argument('--data', help='dataname',
                        default='NBA_traj_position', type=str)
    # parser.add_argument('--data', help='dataname',  default='pkdd-15-predict-taxi-service-trajectory-i', type=str)
    parser.add_argument('--method', help='method',  default='frk1', type=str)
    parser.add_argument('--beta', help='beta',  default=1.0, type=float)
    parser.add_argument('--gamma', help='gamma',  default=1.0, type=float)
    parser.add_argument('--diag_wgt', help='diag_wgt',
                        default=1.0, type=float)
    parser.add_argument(
        '--sample', help='# of sample alignments',  default=100, type=int)
    parser.add_argument(
        '--K', help='# of sample alignments',  default=1, type=int)
    parser.add_argument('--mp', help='use multi processing',
                        default=True, type=bool)

    args = parser.parse_args()

    dirname = './data/' + args.data + '/'
    savedir = dirname + '/grammat/'
    # make savedir
    if not os.path.exists(savedir):
        os.mkdir(savedir)
    print('savedir ', savedir)

    # load data
    fname = dirname + 'data.pkl'
    with open(fname, 'rb') as fobj:
        df = pkl.load(fobj)
    numtraj = df.shape[0]
    print('compute ', numtraj * (numtraj - 1) / 2, 'pairs')

    # scaling locations
    dfmax = df.traj.map(lambda x: x.max()).max()
    dfmin = df.traj.map(lambda x: x.min()).min()
    df.traj = df.traj.map(lambda x: (x-dfmin)/(dfmax-dfmin))

    # construct distance matrix
    result = np.ctypeslib.as_ctypes(np.zeros((numtraj, numtraj)))
    shared_array = sharedctypes.RawArray(result._type_, result)

    # compute gram matrix
    print('gamma', args.gamma)
    inputs = []
    for i in range(numtraj):
        for j in range(i + 1, numtraj):
            if args.mp:
                inputs.append([i, j, args.gamma, args.beta,
                               args.diag_wgt, args.sample, args.K, args.method])
            else:
                run(i, j, args.gamma, args.beta, args.diag_wgt,
                    args.sample, args.K, args.method)
    '''
    if args.mp:
        with Pool(20) as p:
            p.starmap(run, tqdm(inputs))
    '''
    result = np.ctypeslib.as_array(shared_array)
    fname = savedir + args.method + '_gamma_' + str(args.gamma) + '_.csv'
    print('savefile %s' % (fname))
    np.savetxt(fname, result, delimiter=',')
