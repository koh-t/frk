#!/usr/bin/env python3

import pyfrk
import random
import math

random.seed(114514)

N = 10  # Number of sequence
L = 20  # Length of each sequence

# Generate random N sequences
seqs = [[random.random() for i in range(L)] for j in range(N)]

# Parameters for Sampling Alignment Kernel
SAMPLES = 10
BETA = 1.0
DIAG_WGT = 1.0 # Diagonal weight for random walk
SEED = random.randint(0, (1 << 64) - 1) # Random seed for random walk
GAMMA = 1.0 # for e_ij

# Engine of Sampling Alignment Kernel
ker = pyfrk.Kernel(L) # L is the maximum length of sequence considered

for x in range(N):
    for y in range(x, N):
        seq_x = seqs[x]
        seq_y = seqs[y]

        # Make e_ij matrix, i.e., \phi(xi,yj) = exp(-d(xi,yi)/\gamma)
        emat = pyfrk.MatrixF64(len(seq_x), len(seq_y))
        for i, xv in enumerate(seq_x):
            for j, yv in enumerate(seq_y):
                emat.set(i, j, math.exp(-abs(xv - yv)/GAMMA))

        try:
            ans = ker.compute(emat=emat, nsamples=SAMPLES, beta=BETA, diag_wgt=DIAG_WGT, seed=SEED)
            print(f'{x}, {y} => {ans:g}')
        except RuntimeError as e:
            print('Exception:', e)
