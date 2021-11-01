# encoding: utf-8
#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import pickle as pkl
import torch

from sklearn.preprocessing import LabelEncoder


def getmaxmin(df):
    dfmax = df.traj.map(lambda x: x.max(0))
    dfmin = df.traj.map(lambda x: x.min(0))
    lonmax = -1 * np.inf
    latmax = -1 * np.inf
    lonmin = np.inf
    latmin = np.inf
    for i in range(dfmax.shape[0]):
        if lonmax < dfmax[i][0]:
            lonmax = dfmax[i][0]
        if latmax < dfmax[i][1]:
            latmax = dfmax[i][1]
        if lonmin > dfmin[i][0]:
            lonmin = dfmin[i][0]
        if latmin > dfmin[i][1]:
            latmin = dfmin[i][1]
    return lonmax, lonmin, latmax, latmin


def dfnormalize(df, lonmax, lonmin, latmax, latmin):
    min_ = [lonmin, latmin]
    df.traj = df.traj.map(lambda x: x - min_)
    lonw = lonmax - lonmin
    latw = latmax - latmin
    if lonw > latw:
        df.traj = df.traj.map(lambda x: x / lonw)
    else:
        df.traj = df.traj.map(lambda x: x / latw)
    return df


def squared_euclid_pairwise_distance(x, y):
    point1 = x.shape[2]
    point2 = y.shape[2]

    xx = torch.sum(x ** 2, dim=1, keepdim=True)
    yy = torch.sum(y ** 2, dim=1, keepdim=True)
    xx = xx.repeat([1, point2, 1]).transpose(1, 2)
    yy = yy.repeat([1, point1, 1])
    inner = -2 * torch.matmul(x.transpose(2, 1), y)

    dist = xx + inner + yy
    return dist

def get_traj(middir, label_name):
    try:
        df = pd.read_pickle(middir + 'data.pkl')
    except:
        import pickle5
        import pickle
        with open(middir + 'data.pkl', 'rb') as f:
            obj = pickle5.load(f)
        with open(middir + 'data.pkl', 'wb') as f:
            pickle.dump(obj, f)
        df = pd.read_pickle(middir + 'data.pkl')

    lonmax, lonmin, latmax, latmin = getmaxmin(df)
    df = dfnormalize(df, lonmax, lonmin, latmax, latmin)
    if "Omizunagi" in middir:
        if label_name == '':
            df['label'] = df.Sex
        elif label_name == 'month':
            df.Start = pd.to_datetime(df.Start)
            df['label'] = df.Start.dt.month
        elif label_name == 'year':
            df['label'] = df.Year
    if 'GeolifeTrajectories1.3' in middir:
        df = df[df.label.isin(['car', 'taxi'])]
        df = df.reset_index()

    traj = df.traj.to_numpy()
    q = int(df.traj.map(len).median())
    d = df.traj[0].shape[1]
    label, le = enc_label(df)
    print('median length of trajectories is', q)
    return traj, label, le, q, d


def enc_label(df):
    le = LabelEncoder()
    le.fit(df.label.unique())
    label = le.transform(df.label)
    label = torch.tensor(label)
    return label, le