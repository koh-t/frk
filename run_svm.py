# coding: utf-8
import os
import argparse
import numpy as np
import pandas as pd
import pickle as pkl
from tqdm import tqdm
from multiprocessing import Pool

from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import cross_val_score
from sklearn.preprocessing import normalize, LabelEncoder


def pd2label(Y, le):
    # Y = Y.to_numpy()
    Y = le.transform(Y)
    return Y


def evaluate_svm(method, setid, gammas, loaddir, gramdir, svmdir):
    fname = svmdir + method + '_set_' + str(setid) + '.csv'
    if False:  # os.path.exists(fname):
        print("%s exists." % (fname))
    else:
        count = 0
        print("evaluate %s." % (fname))
        for gamma in gammas:
            if count == 0:
                dfres = get_dfres(method, setid, gamma, loaddir, gramdir)
                count += 1
            else:
                dfres = dfres.append(
                    get_dfres(method, setid, gamma, loaddir, gramdir))

        print(dfres[['accuracy', 'f1', 'auc', 'Vaccuracy', 'Vf1', 'Vauc']])
        dirname = svmdir
        fname = dirname + method + '_set_' + str(setid) + '.csv'
        dfres.to_csv(fname, index=False)


def get_dfres(method, setid, gamma, loaddir, gramdir):
    print('method', method, ', gamma', gamma)
    fname = loaddir + 'data.pkl'
    with open(fname, 'rb') as fobj:
        df = pkl.load(fobj)
        if '170309-Omizunagidori-Preprocessed' in loaddir:
            df['label'] = df.Sex
        if 'GeolifeTrajectories1.3' in loaddir:
            id = np.where(df.label.isin(['car', 'taxi']))[0]
            df = df[df.label.isin(['car', 'taxi'])]
            df = df.reset_index()
    # load gram matrix
    fname = gramdir + method + '_gamma_' + str(gamma) + '_.csv'
    R = np.loadtxt(fname, delimiter=',')

    # setinfo
    fname = loaddir + 'setinfo' + str(setid) + '.csv'
    if not os.path.exists(fname):
        ntraj = df.shape[0]
        setinfo = np.zeros(ntraj)
        setinfo[int(ntraj * 0.9):] = 1
        setinfo = np.random.permutation(setinfo)
        np.savetxt(fname, setinfo, delimiter=',')
    else:
        setinfo = np.loadtxt(fname, delimiter=',')

    if R[0, 0] == 0:
        R = R + np.eye(R.shape[0])
    R = normalize(R)

    X = R[setinfo == 0, :]
    X = X[:, setinfo == 0]
    le = LabelEncoder()
    le.fit(df.label.unique())
    Y = pd2label(df.label[setinfo == 0], le)

    Xte = R[setinfo == 1, :]
    Xte = Xte[:, setinfo == 0]
    Yte = pd2label(df.label[setinfo == 1], le)

    clf = SVC(kernel='precomputed', C=1.0, probability=True, max_iter=1000)
    clf.fit(X, Y)
    Ypred = clf.predict(Xte)
    Yprob = clf.predict_proba(Xte)
    print('fit done.')

    acc = accuracy_score(Yte, Ypred)
    rec = recall_score(Yte, Ypred, average='macro')
    pre = precision_score(Yte, Ypred, average='macro')
    f1 = f1_score(Yte, Ypred, average='macro')
    if Yprob.shape[1] == 2:
        auc_ = roc_auc_score(
            Yte, Yprob[:, 1], average='macro', multi_class='ovr')
    else:
        auc_ = roc_auc_score(
            Yte, Yprob, average='macro', multi_class='ovr')
    print('evalucate test score done.')

    ncv = 5
    accv = cross_val_score(
        clf, X, Y, cv=ncv, scoring="accuracy", n_jobs=1).mean()
    recv = cross_val_score(clf, X, Y, cv=ncv, scoring="recall_macro").mean()
    prev = cross_val_score(clf, X, Y, cv=ncv, scoring="precision_macro").mean()
    f1v = cross_val_score(clf, X, Y, cv=ncv, scoring="f1_macro").mean()
    aucv = cross_val_score(clf, X, Y, cv=ncv, scoring="roc_auc_ovr").mean()
    print('cross validation done.')

    _dfres = pd.DataFrame([method, gamma, acc, pre, rec,
                           f1, auc_, accv, prev, recv, f1v, aucv]).T
    _dfres.columns = ['method', 'gamma', 'accuracy', 'precision', 'recall', 'f1', 'auc',
                      'Vaccuracy', 'Vprecision', 'Vrecall', 'Vf1', 'Vauc']

    return _dfres


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='computing grammatrix with kdtw')
    # parser.add_argument('--data', help='dataname',
    #                    default='170309-Omizunagidori-Preprocessed', type=str)
    parser.add_argument('--data', help='dataname',
                        default='NBA_traj_position', type=str)
    parser.add_argument('--method', help='method',  default='frk1', type=str)
    parser.add_argument('--gamma', help='kernel parameter',
                        default='1.0', type=str)
    parser.add_argument('--mp', help='use multi processing',
                        default=False, type=bool)

    args = parser.parse_args()
    args.gamma = [float(x) for x in args.gamma.split(',')]
    loaddir = './data/' + args.data + '/'
    gramdir = './data/' + args.data + '/grammat/'
    svmdir = './data/' + args.data + '/svm/'

    if not os.path.exists(svmdir):
        os.makedirs(svmdir)

    inputs = []
    gammas = args.gamma
    methods = [args.method]
    for setid in range(10):
        for method in methods:
            if args.mp:
                inputs.append(
                    [method, setid, gammas, loaddir, gramdir, svmdir])
            else:
                evaluate_svm(method, setid, gammas, loaddir, gramdir, svmdir)

    if args.mp:
        with Pool(24) as p:
            p.starmap(evaluate_svm, tqdm(inputs))

    print('success.')
