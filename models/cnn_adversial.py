import math
import sys
sys.path.insert(0, './data/')
sys.path.insert(0, './layers/')
sys.path.insert(0, './utils/')

import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import ShuffleSplit
from torch import Tensor

from layers import (BasicConvLayer, BasicDeconvLayer, BasicDenseLayer,
                    get_padding)
from torchsummary import summary
from utils import (get_flat_dim, get_convblock_dim, get_deconvblock_padding)
from cnn import CLASSIFIER, FEAT_EXTRACTOR


class ADVERSIAL_CNN(nn.Module):
    def __init__(self,
                 input_shape=(3, 64, 64),
                 base_filters=[32, 32, 64, 64, 128, 128],
                 kernel_size=[3, 3, 3, 3, 3, 3],
                 stride=[2, 1, 2, 1, 2, 1],
                 n_fc_layers=2,
                 fc_dims=128,
                 output_signers=10,
                 output_classes=10,
                 activation='relu',
                 bnorm=False,
                 dropout=0.5,
                 isTWINS=False,
                 addDense2Conv=False):

        super(ADVERSIAL_CNN, self).__init__()

        self.input_shape = input_shape
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fc_layers = n_fc_layers
        self.fc_dims = fc_dims
        self.output_classes = output_classes
        self.output_signers = output_signers
        self.activation = activation
        self.bnorm = bnorm
        self.dropout = dropout
        self.isTWINS = isTWINS
        self.addDense2Conv = addDense2Conv

        # Initialize feat extraction block
        self.feat_extractor = FEAT_EXTRACTOR(input_shape=self.input_shape,
                                             base_filters=self.base_filters,
                                             kernel_size=self.kernel_size,
                                             stride=self.stride,
                                             activation=self.activation,
                                             bnorm=self.bnorm)

        if self.addDense2Conv:
            # add a dense layer before the classifier
            self.dense = BasicDenseLayer(
                                in_features=self.feat_extractor.flattenDims,
                                out_features=self.fc_dims,
                                bnorm=self.bnorm,
                                activation=self.activation,
                                dropout=self.dropout)

        # Initialize dense block
        self.task_classifier = CLASSIFIER(
                                    input_dims=self.feat_extractor.flattenDims,
                                    output_classes=self.output_classes,
                                    n_layers=self.n_fc_layers,
                                    fc_dims=self.fc_dims,
                                    activation=self.activation,
                                    bnorm=self.bnorm,
                                    dropout=self.dropout,
                                    isTWINS=self.isTWINS,
                                    addDense2Conv=self.addDense2Conv
                                )

        self.adversial_classifier = CLASSIFIER(
                                input_dims=self.feat_extractor.flattenDims,
                                output_classes=self.output_signers,
                                n_layers=self.n_fc_layers,
                                fc_dims=self.fc_dims,
                                activation=self.activation,
                                bnorm=self.bnorm,
                                dropout=self.dropout,
                                isTWINS=self.isTWINS,
                                addDense2Conv=self.addDense2Conv
                            )

    def forward(self, x):
        # feat extraction block
        h = self.feat_extractor(x)

        h = h.reshape(h.size(0), -1)

        # add extra dense layer to conv path?
        if self.addDense2Conv:
            h = self.dense(h)

        # task-specific classification block
        h1_task, y_task = self.task_classifier(h)

        # adversial classification block
        h1_adversial, y_adversial = self.adversial_classifier(h)

        return h1_task, y_task, h1_adversial, y_adversial


if __name__ == '__main__':
    import os
    os.getcwd()
    os.chdir("../")
    os.getcwd()
    print(os.getcwd())

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (3, 100, 100)
    # input_shape = (1024,)
    model = ADVERSIAL_CNN(input_shape=input_shape, addDense2Conv=True).to(device)

    print(model)
    summary(model, input_shape)
    # print(model.flattenDims)
