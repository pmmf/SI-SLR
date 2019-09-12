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


class DENSE_BLOCK(nn.Module):
    def __init__(self,
                 input_dims=1024,
                 dense_dims=[512, 128, 512],
                 activation='relu',
                 bnorm=False,
                 dropout=0.0):

        super(DENSE_BLOCK, self).__init__()

        self.input_dims = input_dims
        self.dense_dims = dense_dims
        self.activation = activation
        self.bnorm = bnorm
        self.dropout = dropout
        self.n_layers = len(self.dense_dims)

        # Initialize encoder layers
        self.create_dense_layers()

    def create_dense_layers(self):
        # first dense layer
        dense_list = nn.ModuleList([
            BasicDenseLayer(in_features=self.input_dims,
                            out_features=self.dense_dims[0],
                            bnorm=self.bnorm,
                            activation=self.activation,
                            dropout=self.dropout)
        ])

        # remaining dense layers
        dense_list.extend([BasicDenseLayer(in_features=self.dense_dims[l-1],
                                           out_features=self.dense_dims[l],
                                           bnorm=self.bnorm,
                                           activation=self.activation,
                                           dropout=self.dropout)
                          for l in range(1, self.n_layers)])

        self.denseBlock = nn.Sequential(*dense_list)

    def forward(self, x):
        # get the activations of each layer
        h_list = ()
        for layer in range(self.n_layers):
            x = self.denseBlock[layer](x)
            h_list += x,
        return h_list


class DEEP_TRANSFER(nn.Module):
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
                 dense_dims=[128],
                 DeconvIsConv=True,
                 hasDecoder=False,
                 hasAdversial=False):

        super(DEEP_TRANSFER, self).__init__()

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
        self.dense_dims = dense_dims
        self.DeconvIsConv = DeconvIsConv
        self.hasDecoder = hasDecoder
        self.hasAdversial = hasAdversial

        # Initialize feat extraction block
        self.feat_extractor = FEAT_EXTRACTOR(input_shape=self.input_shape,
                                             base_filters=self.base_filters,
                                             kernel_size=self.kernel_size,
                                             stride=self.stride,
                                             activation=self.activation,
                                             bnorm=self.bnorm,
                                             get_activations=True)

        # Initialize feat extraction block
        self.dense_block = DENSE_BLOCK(
                            input_dims=self.feat_extractor.flattenDims,
                            dense_dims=self.dense_dims,
                            activation=self.activation,
                            bnorm=self.bnorm,
                            dropout=0.0
                         )

        # Initialize Decoder
        if self.hasDecoder:
            self.decoder = DECODER(input_shape=self.input_shape,
                                   base_filters=self.base_filters,
                                   kernel_size=self.kernel_size,
                                   stride=self.stride,
                                   activation=self.activation,
                                   bnorm=self.bnorm,
                                   DeconvIsConv=self.DeconvIsConv,
                                   fc_dims=self.fc_dims)

        # Initialize dense block
        self.task_classifier = CLASSIFIER(
                                    input_dims=self.dense_dims[-1],
                                    output_classes=self.output_classes,
                                    n_layers=self.n_fc_layers,
                                    fc_dims=self.fc_dims,
                                    activation=self.activation,
                                    bnorm=self.bnorm,
                                    dropout=self.dropout,
                                    isTWINS=self.isTWINS
                                )

        if self.hasAdversial:
            self.adversial_classifier = CLASSIFIER(
                                        input_dims=self.dense_dims[-1],
                                        output_classes=self.output_signers,
                                        n_layers=self.n_fc_layers,
                                        fc_dims=self.fc_dims,
                                        activation=self.activation,
                                        bnorm=self.bnorm,
                                        dropout=self.dropout,
                                        isTWINS=self.isTWINS
                                    )

    def forward(self, x):
        # feat extraction block
        h_conv = self.feat_extractor(x)

        # flatten
        h = h_conv[-1]
        h = h.reshape(h.size(0), -1)

        # dense block
        h_dense = self.dense_block(h)

        # decoder
        if self.hasDecoder:
            x_hat = self.decoder(h_dense[-1])

        # task-specific classification block
        _, y_task = self.task_classifier(h_dense[-1])

        # adversial classification block
        if self.hasAdversial:
            _, y_adversial = self.adversial_classifier(h_dense[-1])

        # return outputs
        if self.hasDecoder:
            return h_conv, h_dense, y_task, x_hat
        elif self.hasAdversial:
            return h_conv, h_dense, y_task, y_adversial
        else:
            return h_conv, h_dense, y_task


class DECODER(nn.Module):
    def __init__(self,
                 input_shape=(3, 64, 64),
                 base_filters=[64, 128, 256, 512],
                 kernel_size=[3, 3, 3, 3],
                 stride=[2, 2, 2, 2],
                 activation='leaky_relu',
                 fc_dims=128,
                 bnorm=True,
                 DeconvIsConv=True):

        super(DECODER, self).__init__()

        self.input_shape = input_shape
        self.base_filters = base_filters
        self.kernel_size = kernel_size
        self.stride = stride
        self.activation = activation
        self.bnorm = bnorm
        self.DeconvIsConv = DeconvIsConv
        self.fc_dims = fc_dims

        # Initialize decoder layers
        self.create_decoder_layers()

    def create_decoder_layers(self):
        # feature maps size and number of filters of the first layer
        hConvOut, wConvOut = get_convblock_dim(self.input_shape,
                                               self.base_filters,
                                               self.kernel_size, self.stride)

        self.feat_sze_h, self.feat_sze_w = hConvOut[-1], wConvOut[-1]
        hConvOut.pop(-1)
        wConvOut.pop(-1)

        # if denconv is not transpose of conv
        deconv_index = -1
        if not self.DeconvIsConv:
            deconv_index -= 1

        self.n_filt = self.base_filters[deconv_index]

        # first dense
        out_features = self.feat_sze_h*self.feat_sze_w*self.n_filt
        self.dense = BasicDenseLayer(in_features=self.fc_dims,
                                     out_features=out_features,
                                     activation=self.activation,
                                     bnorm=self.bnorm)
        # Deconv layers
        # reserve net architecture lists
        income_channels = self.base_filters[deconv_index::-1]
        income_channels.insert(0, self.n_filt)
        income_channels.pop(-1)
        outcome_channels = self.base_filters[deconv_index::-1]
        reverse_kernel = self.kernel_size[deconv_index::-1]
        reverse_stride = self.stride[deconv_index::-1]
        hConvOut = hConvOut[deconv_index::-1]
        wConvOut = wConvOut[deconv_index::-1]

        # get output paddings
        out_pad = get_deconvblock_padding((self.feat_sze_h, self.feat_sze_w),
                                          reverse_kernel, reverse_stride,
                                          (hConvOut, wConvOut))

        # deconv layers
        deconv_list = nn.ModuleList([
            BasicDeconvLayer(in_channels=income_channels[l],
                             out_channels=outcome_channels[l],
                             activation=self.activation,
                             bnorm=self.bnorm,
                             output_padding=out_pad[l][0],
                             kernel_size=reverse_kernel[l],
                             stride=reverse_stride[l])
            for l in range(len(income_channels))])

        # output layer - NO BN!
        deconv_list.extend([BasicDeconvLayer(in_channels=outcome_channels[-1],
                                             out_channels=self.input_shape[0],
                                             kernel_size=self.kernel_size[0],
                                             stride=1,
                                             output_padding=0,
                                             bnorm=False,
                                             activation='tanh')])

        # DeconvBlock
        self.DenconvBlock = nn.Sequential(*deconv_list)

    def forward(self, h):
        h = self.dense(h)
        h = h.view(-1, self.n_filt, self.feat_sze_h, self.feat_sze_w)
        h = self.DenconvBlock(h)
        return h


if __name__ == '__main__':
    import os
    # os.getcwd()
    # os.chdir("../")
    # os.getcwd()
    # print(os.getcwd())

    os.environ['CUDA_VISIBLE_DEVICES'] = '2'
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    input_shape = (3, 100, 100)

    model = DEEP_TRANSFER(input_shape=input_shape).to(device)

    print(model)
    summary(model, input_shape)
