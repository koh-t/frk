# encoding: utf-8
# !/usr/bin/env python3
import os
import sys
import difflib
import shutil
import argparse
import numpy as np
import pickle as pkl
from tqdm import tqdm
from collections import OrderedDict

import torch
import torchvision
from torch import nn
from torch import optim
import torch.nn.functional as F
import torch.nn.utils.rnn as rnn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

from scipy.interpolate import BSpline
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc, roc_curve, accuracy_score, precision_score, recall_score, f1_score

import sample_alignments as sa
from util import getmaxmin, dfnormalize, squared_euclid_pairwise_distance, get_traj, enc_label


def smooth_min(phi, beta):
    exp_phi = torch.exp(-phi / beta)
    partition = torch.sum(exp_phi, 1, keepdim=True)
    odds = torch.mean(phi * exp_phi / partition, 1, keepdim=True)
    # odds = torch.sum(phi * exp_phi / partition, 1, keepdim=True)
    odds = odds.transpose(0, 1)
    return odds


def sample_trajectory(traj, d, q, Ctraj=10, grad=False, frechet=[], sampling='random'):
    if sampling == 'random':
        # Random Sampling
        print('Random Sampling')
        seedidx = np.random.permutation(len(traj))[:Ctraj]
        trajseed = traj[seedidx]

        Y = np.zeros([Ctraj, d, q])
        for i in range(Ctraj):
            traj = trajseed[i]
            x = np.arange(q)
            xp = np.arange(len(traj)) / (len(traj) + 1) * q
            for j in range(2):
                Y[i, j, :] = np.interp(x, xp, traj[:, j])

    else:
        # Furthest Point Sampling
        seedidx = np.random.permutation(len(traj))[:Ctraj]
        print('Furthest Point Sampling with %s' % types)

        frechet += frechet.T
        frechet /= frechet.max()
        if sampling == 'len' or sampling == 'lenfre':
            trajlen = np.array(list(map(len, traj)))
            lenmat = np.zeros([len(traj), len(traj)])
            for i in range(len(traj)):
                for j in range(i, len(traj)):
                    lenmat[i, j] = np.abs(trajlen[i] - trajlen[j])
            lenmat += lenmat.T
            lenmat /= lenmat.max()
        if sampling == 'len':
            mat = lenmat
        elif sampling == 'lenfre':
            mat = lenmat * frechet
        else:
            mat = frechet

        traj_inuse = []
        traj_notuse = np.arange(len(traj)).tolist()
        traj_inuse.append(seedidx[0])
        traj_notuse.pop(seedidx[0])
        for i in range(Ctraj - 1):
            db = mat[traj_inuse, :]
            db = db[:, traj_notuse]
            db = db.min(0)
            quenew = traj_notuse[db.argmax()]
            traj_inuse.append(quenew)
            traj_notuse.pop(traj_notuse.index(quenew))

        Y = np.zeros([Ctraj, d, q])
        for i in range(Ctraj):
            _traj = traj[traj_inuse[i]]
            x = np.arange(q + 1)
            xp = np.arange(len(_traj)) / (len(_traj) + 1) * q
            for j in range(2):
                spl = BSpline(xp, _traj[:, j], 2)
                xi = spl(x)
                # Y[i, j, :] = np.interp(x, xp, _traj[:, j])

    if grad:
        Y = torch.tensor(Y, requires_grad=True).float()
    else:
        Y = torch.tensor(Y, requires_grad=False).float()

    return Y


class Net1(nn.Module):
    def __init__(self, traj, d, q=32, Ctraj=10, scale=1e-4, C=2, frechet=[], sampling='random', dropout=0.1, seed=1,
                 args=''):
        # Linear
        super(Net1, self).__init__()
        self.q = q
        self.d = d
        self.Ctraj = Ctraj
        self.scale = scale
        self.beta = args.beta
        # nn.Parameter(torch.tensor([1e-5], requires_grad=True))
        # Y is a set of random trajectories
        np.random.seed(seed)
        torch.manual_seed(seed)
        self.Y = sample_trajectory(traj, d, q, Ctraj, False, frechet)
        self.fc = nn.Linear(Ctraj, C)
        self.dp = nn.Dropout(dropout)
        self.device = args.device

    def forward(self, _traj, Aset):
        # project Y
        Y = self.Y.permute([2, 0, 1]).to(device=self.device)
        Y = Y.permute([1, 2, 0])

        for i in range(len(Aset)):
            A = Aset[i, :]
            X = torch.tensor(_traj[i]).view(_traj[i].shape[0], 1, -1).float().to(device=self.device)
            X = X.permute([1, 2, 0])
            D = squared_euclid_pairwise_distance(X, Y)
            # D = self.dp(D)

            # frechet kernel
            _phi = torch.exp(-D / self.scale)
            _odds = 0
            for _A in A:
                phi = _phi[:, _A[:, 0], _A[:, 1]]
                _odds += smooth_min(phi, self.beta)
            # _odds /= len(A)

            # mlp
            _odds = self.fc(_odds)
            _odds = F.selu(_odds)
            # _odds = self.dp(_odds)

            if i == 0:
                odds = _odds
            else:
                odds = torch.cat([odds, _odds])
        return odds


class Net2(nn.Module):
    def __init__(self, traj, d, q=32, Ctraj=10, scale=1e-4, C=2, frechet=[], sampling='random', dropout=0.1, seed=1,
                 args=''):
        # Linear + RNN + ReLU
        super(Net2, self).__init__()
        self.q = q
        self.d = d
        self.Ctraj = Ctraj
        self.scale = scale
        self.beta = args.beta
        # nn.Parameter(torch.tensor([1e-5], requires_grad=True))
        # Y is a set of random trajectories
        np.random.seed(seed)
        torch.manual_seed(seed)
        # self.Y = sample_trajectory(traj, d, q, Ctraj, False, frechet, sampling=sampling)
        self.lstm = nn.LSTM(d, q, dropout=dropout)
        self.fc = nn.Linear(q, C)
        self.dp = nn.Dropout(dropout)
        self.device = args.device

    def forward(self, _traj, Aset):
        # project Y
        hidden = (
            torch.randn(1, self.Ctraj, self.d).to(device=self.device),
            torch.randn(1, self.Ctraj, self.d).to(device=self.device))
        Y = self.Y.permute([2, 0, 1]).to(device=self.device)
        Y, hidden = self.lstm(Y, hidden)
        Y = Y.permute([1, 2, 0])

        for i in range(len(Aset)):
            A = Aset[i, :]
            X = torch.tensor(_traj[i]).view(_traj[i].shape[0], 1, -1).float().to(device=self.device)
            hidden = (
                torch.randn(1, 1, self.d).to(device=self.device), torch.randn(1, 1, self.d).to(device=self.device))
            X, hidden = self.lstm(X, hidden)
            X = X.permute([1, 2, 0])
            D = squared_euclid_pairwise_distance(X, Y)
            # D = self.dp(D)

            # frechet kernel
            _phi = torch.exp(-D / self.scale)
            _odds = 0
            for _A in A:
                phi = _phi[:, _A[:, 0], _A[:, 1]]
                _odds += smooth_min(phi, self.beta)
            # _odds /= len(A)

            # mlp
            _odds = self.fc(_odds)
            _odds = F.selu(_odds)
            # _odds = self.dp(_odds)

            if i == 0:
                odds = _odds
            else:
                odds = torch.cat([odds, _odds])
        return odds

class LSTM1(nn.Module):
    def __init__(self, traj, d, q=32, Ctraj=10, scale=1e-4, C=2, frechet=[], sampling='random', dropout=0.1, seed=1,
                 args=''):
        # Linear + RNN + ReLU
        super(LSTM1, self).__init__()
        self.q = q
        self.d = d
        self.Ctraj = Ctraj
        self.scale = scale
        self.beta = args.beta
        # nn.Parameter(torch.tensor([1e-5], requires_grad=True))
        # Y is a set of random trajectories
        np.random.seed(seed)
        torch.manual_seed(seed)
        # self.Y = sample_trajectory(traj, d, q, Ctraj, False, frechet, sampling=sampling)
        self.lstm = nn.LSTM(input_size=d,
         hidden_size=q, dropout=dropout,
         batch_first=True, bidirectional=True, num_layers=1)
        # self.fc1 = nn.Linear(d, d)
        # self.fc2 = nn.Linear(d, q)
        self.fc = nn.Linear(q, C)
        self.dp = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(q)
        self.device = args.device

    def forward(self, _traj, Aset):
        # project Y
        traj_len = torch.LongTensor([len(__traj) for __traj in _traj]).to(device=self.device)
        X = rnn.pad_sequence(_traj, batch_first=True).to(device=self.device)
        # X = F.relu(self.fc1(X))
        # X = F.selu(self.fc2(X))
        hidden = (
            torch.randn(2, X.shape[0], self.q).to(device=self.device), 
            torch.randn(2, X.shape[0], self.q).to(device=self.device))
        X = pack_padded_sequence(X, traj_len, batch_first=True, enforce_sorted=False)
        X, (ht, ct) = self.lstm(X, hidden)
        # X, output_lengths = pad_packed_sequence(X, batch_first=True)
        X = ht[-1]
        X = self.bn(X)
        X = self.fc(X)
        # X = F.softmax(X, dim=1)
        # X = [X[torch.LongTensor(i).to(device=self.device), id, :] for (i, id) in enumerate(traj_len)]
        return X


class GRU1(nn.Module):
    def __init__(self, traj, d, q=32, Ctraj=10, scale=1e-4, C=2, frechet=[], sampling='random', dropout=0.1, seed=1,
                 args=''):
        # Linear + RNN + ReLU
        super(GRU1, self).__init__()
        self.q = q
        self.d = d
        self.Ctraj = Ctraj
        self.scale = scale
        self.beta = args.beta
        # nn.Parameter(torch.tensor([1e-5], requires_grad=True))
        # Y is a set of random trajectories
        np.random.seed(seed)
        torch.manual_seed(seed)
        # self.Y = sample_trajectory(traj, d, q, Ctraj, False, frechet, sampling=sampling)
        self.lstm = nn.GRU(input_size=d,
         hidden_size=q, dropout=dropout,
         batch_first=True, bidirectional=True, num_layers=1)
        # self.fc1 = nn.Linear(d, d)
        # self.fc2 = nn.Linear(d, q)
        self.fc = nn.Linear(q, C)
        self.dp = nn.Dropout(dropout)
        self.bn = nn.BatchNorm1d(q)
        self.device = args.device

    def forward(self, _traj, Aset):
        # project Y
        traj_len = torch.LongTensor([len(__traj) for __traj in _traj]).to(device=self.device)
        X = rnn.pad_sequence(_traj, batch_first=True).to(device=self.device)
        # X = F.relu(self.fc1(X))
        # X = F.selu(self.fc2(X))
        hidden = torch.randn(2, X.shape[0], self.q).to(device=self.device)
        X = pack_padded_sequence(X, traj_len, batch_first=True, enforce_sorted=False)
        X, (ht) = self.lstm(X, hidden)
        # X, output_lengths = pad_packed_sequence(X, batch_first=True)
        X = ht[-1]
        X = self.bn(X)
        X = self.fc(X)
        # X = F.softmax(X, dim=1)
        # X = [X[torch.LongTensor(i).to(device=self.device), id, :] for (i, id) in enumerate(traj_len)]
        return X

