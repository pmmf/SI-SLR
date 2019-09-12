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


class CNN(nn.Module):
    def __init__(self,
                 input_shape=(3, 64, 64),
                 base_filters=[32, 32, 64, 64, 128, 128],
                 kernel_size=[3, 3, 3, 3, 3, 3],
                 stride=[2, 1, 2, 1, 2, 1],
                 n_fc_layers=2,
                 fc_dims=128,
                 output_classes=10,
                 activation='relu',
                 bnorm=False,
                 dropout=0.5,
                 isTWINS=False):

        super(CNN, self).__init__()

        self.input_shape = input_shape
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_fc_layers = n_fc_layers
        self.fc_dims = fc_dims
        self.output_classes = output_classes
        self.activation = activation
        self.bnorm = bnorm
        self.dropout = dropout
        self.isTWINS = isTWINS

        # Initialize feat extraction block
        self.feat_extractor = FEAT_EXTRACTOR(input_shape=self.input_shape,
                                             base_filters=self.base_filters,
                                             kernel_size=self.kernel_size,
                                             stride=self.stride,
                                             activation=self.activation,
                                             bnorm=self.bnorm)

        # Initialize dense block
        self.classifier = CLASSIFIER(
                                    input_dims=self.feat_extractor.flattenDims,
                                    output_classes=self.output_classes,
                                    n_layers=self.n_fc_layers,
                                    fc_dims=self.fc_dims,
                                    activation=self.activation,
                                    bnorm=self.bnorm,
                                    dropout=self.dropout,
                                    isTWINS=self.isTWINS
                                )

    def forward(self, x):
        # feat extraction block
        x = self.feat_extractor(x)

        # check if flatten is needed
        if len(x.size()) > 2:
            x = x.reshape(x.size(0), -1)

        # classification block (dense)
        h1, x = self.classifier(x)

        return h1, x


class CLASSIFIER(nn.Module):
    def __init__(self,
                 input_dims=1024,
                 output_classes=10,
                 n_layers=3,
                 fc_dims=128,
                 activation='relu',
                 bnorm=False,
                 dropout=0.5,
                 isTWINS=False,
                 addDense2Conv=False):

        super(CLASSIFIER, self).__init__()

        self.input_dims = input_dims
        self.output_classes = output_classes
        self.n_layers = n_layers
        self.fc_dims = fc_dims
        self.activation = activation
        self.bnorm = bnorm
        self.dropout = dropout
        self.isTWINS = isTWINS
        self.addDense2Conv = addDense2Conv

        # Initialize encoder layers
        self.create_dense_layers()

    def create_dense_layers(self):
        # Conv layers
        # first dense layer
        first_dropout = self.dropout
        if self.isTWINS:
            first_dropout = 0.0

        if self.addDense2Conv:
            self.input_dims = self.fc_dims

        dense_list = nn.ModuleList([
            BasicDenseLayer(in_features=self.input_dims,
                            out_features=self.fc_dims,
                            bnorm=self.bnorm,
                            activation=self.activation,
                            dropout=first_dropout)
        ])

        # remaining dense layers
        dense_list.extend([BasicDenseLayer(in_features=self.fc_dims,
                                           out_features=self.fc_dims,
                                           bnorm=self.bnorm,
                                           activation=self.activation,
                                           dropout=self.dropout)
                          for l in range(1, self.n_layers)])

        # Last dense layer
        dense_list.append(BasicDenseLayer(in_features=self.fc_dims,
                                          out_features=self.output_classes,
                                          bnorm=self.bnorm,
                                          activation='linear')
                          )

        self.denseBlock = nn.Sequential(*dense_list)

    def forward(self, x):
        # conv blocks
        h1 = self.denseBlock[0](x)
        x = self.denseBlock[1:](h1)
        return h1, x


class FEAT_EXTRACTOR(nn.Module):
    def __init__(self,
                 input_shape=(3, 64, 64),
                 base_filters=[32, 32, 64, 64, 128, 128],
                 kernel_size=[3, 3, 3, 3, 3, 3],
                 stride=[2, 1, 2, 1, 2, 1],
                 activation='relu',
                 bnorm=False,
                 get_activations=False):

        super(FEAT_EXTRACTOR, self).__init__()

        self.input_shape = input_shape
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.bnorm = bnorm
        self.get_activations = get_activations

        # Initialize encoder layers
        self.create_encoder_layers()

        # Get flatten dim
        self.flattenDims = self.get_flatten_dim()

    def get_flatten_dim(self):
        return get_flat_dim(self.input_shape, self.base_filters,
                            self.kernel_size, self.stride)

    def create_encoder_layers(self):
        # Conv layers
        # first conv layer
        conv_list = nn.ModuleList([
            BasicConvLayer(in_channels=self.input_shape[0],
                           out_channels=self.base_filters[0],
                           bnorm=self.bnorm,
                           activation=self.activation,
                           kernel_size=self.kernel_size[0],
                           stride=self.stride[0])
        ])

        # remaining conv layers
        conv_list.extend([BasicConvLayer(in_channels=self.base_filters[l-1],
                                         out_channels=self.base_filters[l],
                                         bnorm=self.bnorm,
                                         activation=self.activation,
                                         kernel_size=self.kernel_size[l],
                                         stride=self.stride[l])
                         for l in range(1, len(self.base_filters))])

        # ConvBlock
        self.ConvBlock = nn.Sequential(*conv_list)

    def forward(self, x):
        # conv blocks
        if not self.get_activations:
            x = self.ConvBlock(x)
            return x
        else:
            h_list = ()
            for layer in range(len(self.base_filters)):
                x = self.ConvBlock[layer](x)
                h_list += x,
            return h_list


if __name__ == '__main__':
    import os
    os.getcwd()
    os.chdir("../")
    os.getcwd()
    print(os.getcwd())

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (3, 100, 100)
    # input_shape = (1024,)
    model = CNN(input_shape=input_shape).to(device)

    print(model)
    summary(model, input_shape)
    # print(model.flattenDims)
