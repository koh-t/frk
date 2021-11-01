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
  - `compute_with_topk(emat,tf,nsamples,beta,diag_wgt,seed)` computes the kernel value from a distance matrix `emat=pyfrk.MatrixF64` and topK results `tf=pyfrk.TopkFrechet` (i.e., FRK2).
- `pyfrk.TopkFrechet` computes the topK Frechet distances.
  - `__init__(emat)` initilizes the class using the distance matrix in O(nrows*ncols) time.
  - `next()` incrementally computes the topK Frechet distances in O(nrows*ncols) time.

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
```

## Example (FRK2)

`sample/sample_topk.py`

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

# Parameters for TopK Sampling Alignment Kernel
K = 3
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
            # Compute top-K points on Frechet
            tf = pyfrk.TopkFrechet(emat)
            for k in range(K):
                # if there is no next point, will return False
                if not tf.next():
                    break
            ans = ker.compute_with_topk(emat=emat, tf=tf, nsamples=SAMPLES, beta=BETA, diag_wgt=DIAG_WGT, seed=SEED)
            print(f'{x}, {y} => {ans:f}')

            # If you want to continue experimenting with large K, perform tf.next() more
            # for k in range(2): # K+2
            #     if not tf.next():
            #         break
            # ans = ker.compute_with_topk(emat=emat, tf=tf, nsamples=SAMPLES, beta=BETA, diag_wgt=DIAG_WGT, seed=SEED)
            # print(f'{x}, {y} => {ans:f}')
        except RuntimeError as e:
            print('Exception:', e)
```

## Benchmark for time performance

`src/timeperf.cpp` provides the benchmark for time performances of some algorithms.

### Build instructions

```shell
$ mkdir build
$ cd build
$ cmake ..
$ make
```

### Running example

```shell
$ ./timeperf
usage: ./timeperf --dirpath=string [options] ... 
options:
  -i, --dirpath       directory path of trajectories (in csv) (string)
  -o, --outpath       output file path of detailed results (in csv) (string [=out.csv])
  -a, --algo          algorithm (frk | frk2 | ga | dtw | frec | haus) (string [=frk])
  -k, --nsamples      #samples (in frk) (unsigned long [=100])
  -K, --topK          topK evaluated (in frk2) (unsigned long [=3])
  -b, --beta          beta (in frk) (double [=1])
  -w, --diag_wgt      weight of diagonal walk (in frk) (double [=1])
  -s, --seed          seed of random walk (in frk) (unsigned long [=114514])
  -g, --sigma         sigma (in ga) (double [=2])
  -t, --triangular    triangular (in ga) (int [=0])
  -?, --help          print this message
```

The command reads all csv files of trajectories at `dirpath` and evaluates all the pairs of trajectories. The detailed benchmark results will output in `outpath` as a csv file.

#### Example: Evaluation for FRK1

```shell
$ ./timeperf -i trajs -o out.csv -a frk
num_trajs: 20
max_length: 380
preprocess_time_in_ms: 28.830111
average_compute_time_in_ms: 4.326352
$ head out.csv
traj_x,traj_y,compute_ms,ans
traj0.csv,traj0.csv,2.673531,0.680515
traj0.csv,traj1.csv,2.625256,0.057540
traj0.csv,traj2.csv,4.307953,0.087399
traj0.csv,traj3.csv,6.205411,0.021527
traj0.csv,traj4.csv,2.865531,0.042582
traj0.csv,traj5.csv,3.939732,0.022058
traj0.csv,traj6.csv,3.645571,0.010842
traj0.csv,traj7.csv,2.603868,0.004307
traj0.csv,traj8.csv,2.950798,0.035579
```
