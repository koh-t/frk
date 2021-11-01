# This is a set of scripts reproducing the experimental results of "Fr{\'e}chet Kernel for Trajectory Analysis"
First, please follow frk_anonymous/README.md to install our C++ implementations of FRK.

1. To reproduce results of SVMs in Section 4, run frk_mp.py or *_mp.py to calcurate gram matrices for each kernel or distance. Then, run run_svm.py to get scores.
For RNNs, run demo_lstm.py or demo_gru.py

Preparation:
# pyfrk

This library is a C++11 implementation of Frechet Kernel (FRK). You can build a Python module using Pybind11.

## Install

The library builds a Python module from `src/frk.cpp` using [Pybind11](https://github.com/pybind/pybind11), so you have to install Pybind11. Following the [official document](https://pybind11.readthedocs.io/en/master/basics.html), you can install it from the source files as follows.

```shell
git clone https://github.com/pybind/pybind11.git
cd pybind11
mkdir build
cd build
cmake ..
make check -j4
make install
```

Then, you can build `pyfrk` module through  `setup.py` in the following command.

```shell
pip install .
```

If necessary, set the include path for Pybind11 on line 19 of `setup.py`.

## Usage

`pyfrk` module has the following classes and functions:

- `pyfrk.MatrixF64`: A matrix of Float64.
  - `__init__(nrows, ncols)` reserves the matrix size.
  - `set(i,j,e)` sets value `e` in row `0<=i<nrows`, column `0<=j<ncols` of the matrix.
- `pyfrk.Kernel`: Class for computing Frechet Kernel values.
  - `__init__(maxsize)` initializes the class using the maximum length of input trajectories, taking O(`maxsize`^2) time.
  - `compute(emat,nsamples,beta,diag_wgt,seed)` computes the kernel value from a distance matrix `emat=pyfrk.MatrixF64` (i.e., FRK1).
    - `nsamples` is the number of samples, `beta` is a parameter for smooth-min function, `diag_wgt` is a weight of the probability of transition to the diagonal, `seed` is a seed value for uniform distribution. Setting `diag_wgt=1.0` indicates random sampling on uniform distribution. Setting `diag_wgt>1.0` increases the probability of transition to the diagonal.
  
## Example (FRK1)

`sample/sample.py`

```python
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
