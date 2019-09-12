import sys
sys.path.insert(0, './layers/')

import math
from layers import get_padding
import numpy as np
import torch
import matplotlib.pyplot as plt
plt.switch_backend('agg')
import copy
from  sklearn.metrics import silhouette_score


def one_hot_1D(n_classes, label):
    one_hot = torch.zeros(n_classes,)
    one_hot[label] = 1
    return one_hot


def one_hot_2D(n_classes, size, label):
    one_hot = torch.zeros(n_classes, size[0], size[1])
    one_hot[label, :, :] = 1
    return one_hot


def dunn_index(X, clusters):
    """
    Inputs:
    X - np.array of size (N, D)
    clusters - np.array of size (N), where each element is an integer in the
        set {0, ..., C-1}

    Outputs:
    d - np.array of size (C)
    """
    D = X.shape[1]
    C = np.amax(clusters) + 1
    d = np.zeros(C)
    for c in range(C):
        Xc = X[clusters == c, :]
        Xo = X[clusters != c, :]

        Nc = Xc.shape[0]
        No = Xo.shape[0]
        inter_dists = np.linalg.norm(Xc.reshape(Nc, 1, D) -
                                     Xo.reshape(1, No, D), axis=2)
        intra_dists = np.linalg.norm(Xc.reshape(Nc, 1, D) -
                                     Xc.reshape(1, Nc, D), axis=2)

        min_sep = np.amin(inter_dists)
        max_diam = np.amax(intra_dists)

        d[c] = min_sep/max_diam

    return d


def mean_silhouette_index(X, clusters):
    """
    Inputs:
    X - np.array of size (N, D)
    clusters - np.array of size (N), where each element is an integer in the
        set {0, ..., C-1}

    Outputs:
    s - np.array of size (C)
    """
    D = X.shape[1]
    C = np.amax(clusters) + 1
    s = np.zeros(C)
    for c in range(C):
        Xc = X[clusters == c, :]
        Xo = X[clusters != c, :]

        Nc = Xc.shape[0]
        No = Xo.shape[0]
        a = np.sum(np.linalg.norm(Xc.reshape(Nc, 1, D) -
                                  Xc.reshape(1, Nc, D), axis=2), axis=1)/(Nc-1)
        b = np.sum(np.linalg.norm(Xc.reshape(Nc, 1, D) -
                                  Xo.reshape(1, No, D), axis=2), axis=1)/No
        s_per_sample = (b - a)/np.maximum(a, b)
        s[c] = np.mean(s_per_sample)

    return s


def signer_invariance_metrics(X, y, s):
    """
    Inputs:
    X - np.array of size (N, D)
    y - np.array of size (N)
    s - np.array of size (N)

    Outputs:
    d_sign - np.array of size (n_signs)
    d_signer - np.array of size (n_signs, n_signers)

    dict_sign - dictionary mapping sign indices to sign labels
    dict_signer - dictionary mapping signer indices to signer labels
    """
    y_set = np.unique(y)
    s_set = np.unique(s)

    n_signs = len(y_set)
    n_signers = len(s_set)

    dict_sign = {y_set[i]: i for i in range(n_signs)}
    dict_signer = {s_set[i]: i for i in range(n_signers)}

    y_norm = np.zeros_like(y, dtype=int)
    for i in range(1, n_signs):
        y_norm[y == y_set[i]] = i

    s_norm = np.zeros_like(s, dtype=int)
    for i in range(1, n_signers):
        s_norm[s == s_set[i]] = i

    d_sign = dunn_index(X, y_norm)
    sil_sign = mean_silhouette_index(X, y_norm)
    # sil_sign = silhouette_score(X, y_norm)

    d_signer = np.zeros((n_signs, n_signers))
    sil_signer = np.zeros((n_signs, n_signers))
    for i, sign in enumerate(y_set):
        si = s_norm[y == sign]
        Xi = X[y == sign, :]
        d_signer[i] = dunn_index(Xi, si)
        sil_signer[i] = np.mean(mean_silhouette_index(Xi, si))
        # sil_signer[i] = silhouette_score(Xi, si)

    return d_sign, d_signer, sil_sign, sil_signer, dict_sign, dict_signer


def get_evaluation_protocol(dataset):
    """get evaluation protocol depending of the dataset"""

    if dataset == 'psl':
        IM_SIZE = (3, 64, 64)
        MODE = [1, 2, 3, 5, 7, 9, 10, 11]
        SPLITS = 1
        n_classes = 31
        n_signers = 6
    elif dataset == 'triesch':
        IM_SIZE = (1, 64, 64)
        MODE = [0, 1, 2, 3, 4, 5, 6, 7]
        SPLITS = 1
        n_classes = 10
        n_signers = 7
    elif dataset == 'staticSL':
        IM_SIZE = (3, 100, 100)
        MODE = 'groups'
        SPLITS = 5
        n_classes = 10
        n_signers = 10

    return IM_SIZE, MODE, SPLITS, n_signers, n_classes


def inverse_transform(x, isInGPU=True):
    if isInGPU:
        x = x.to('cpu').numpy()
    else:
        x = x.numpy()

    if len(x.shape) == 4:
        x = x.transpose((0, 2, 3, 1))
    elif len(x.shape) == 3:
        x = x.transpose((1, 2, 0))

    x = (x+1.)/2.
    return x


def get_deconvblock_padding(input_dim, kernel_sizes, strides, out_dims):
    ''' Get size of each deconv layer output '''
    H_in, W_in = input_dim
    out_H, out_W = out_dims
    out_pad = []

    for i in range(len(kernel_sizes)):
        # H_out
        H = int((H_in-1)*strides[i] - 2*get_padding(kernel_sizes[i]) +
                kernel_sizes[i])
        # out_padding = H - H_desired
        out_h = out_H[i]-H
        # update H_in
        H_in = H + out_h

        # W_out
        W = int((W_in-1)*strides[i] - 2*get_padding(kernel_sizes[i]) +
                kernel_sizes[i])
        # out_padding = W - W_desired
        out_w = out_W[i]-W
        # update W_in
        W_in = W + out_w

        # Append output paddings
        out_pad.append((out_h, out_w))

    return out_pad


def get_convblock_dim(input_dim, conv_filters, kernel_sizes, strides):
    ''' Get size of each conv layer output '''
    _, H, W = input_dim
    h_list = [H]
    w_list = [W]
    for i in range(len(conv_filters)):
        H = int(math.floor((H + 2*get_padding(kernel_sizes[i]) -
                kernel_sizes[i])/(1.*strides[i]) + 1))
        h_list.append(H)

        W = int(math.floor((W + 2*get_padding(kernel_sizes[i]) -
                kernel_sizes[i])/(1.*strides[i]) + 1))
        w_list.append(W)

    return h_list, w_list


def get_flat_dim(input_dim, conv_filters, kernel_sizes, strides):
    ''' Get flatten dimension after conv block '''
    _, H, W = input_dim
    for i in range(len(conv_filters)):
        H = int(math.floor((H + 2*get_padding(kernel_sizes[i]) -
                kernel_sizes[i])/(1.*strides[i]) + 1))
        W = int(math.floor((W + 2*get_padding(kernel_sizes[i]) -
                kernel_sizes[i])/(1.*strides[i]) + 1))

    flat_dim = H * W * conv_filters[-1]

    return flat_dim


if __name__ == '__main__':
    pass