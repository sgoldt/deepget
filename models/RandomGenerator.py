#!/usr/bin/env python3
#
# Random fully connected generator, and a fully connected Autoencoder.
#
# Date: April 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

from collections import OrderedDict
import math

import torch
import torch.nn as nn


class Sign(nn.Module):
    r"""Applies the sign function  element-wise:

    Args:
        inplace: can optionally do the operation in-place. Default: ``False``

    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    """
    __constants__ = ["inplace"]

    def __init__(self, inplace=False):
        super(Sign, self).__init__()
        self.inplace = inplace

    def forward(self, input):
        return torch.sign(input, out=input) if self.inplace else torch.sign(input)

    def extra_repr(self):
        inplace_str = "inplace=True" if self.inplace else ""
        return inplace_str


class RandomGenerator(nn.Module):
    """
    A deep generator of input dimension D, output dimension N, depth L, with
    activation function g between intermediate layers and a tanh activation at
    the final layer.
    """

    def __init__(self, Ds, activation, batchnorm=True):
        """
        Parameters:
        -----------

        Ds :
            array containing the number of nodes of all the layers of the generator,
            with the latent dimension = Ds[0] and the output dimension = Ds[-1].
        activation : (nn.Module)
            an array with the activation functions of the network
        batchnorm :
            if true, add batchnorm layers before the hidden layers
        """
        super(RandomGenerator, self).__init__()
        self.generator = None

        self.num_layers = len(Ds) - 1
        if self.num_layers == 0:
            self.generator = nn.Sequential(nn.Tanh())
        else:
            layers = []
            # define the intermediate generator layers
            for l in range(self.num_layers):
                N_in = Ds[l]
                N_out = Ds[l + 1]
                g = activation
                layers += [("w%d" % l, nn.Linear(N_in, N_out, bias=False))]
                if batchnorm:
                    # add a batch norm layer that does nothing smart,
                    # just recenters the variables x = (x - E x) / sqrt(Var x^2)
                    bn = nn.BatchNorm1d(
                        N_out, affine=False, momentum=None, track_running_stats=False,
                    )
                    layers += [("b%d" % l, bn)]
                layers += [("g%d" % l, g())]
            # build the generator
            self.generator = nn.Sequential(OrderedDict(layers))

        self.generator.eval()
        # initialise the weights of the layers
        for param in self.parameters():
            nn.init.normal_(param.data, 0, 1.0 / math.sqrt(param.shape[1]))
            param.requires_grad = False

    def forward(self, input):
        return torch.sign(self.generator(input))


class AutoEncoder(nn.Module):
    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(28 * 28, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(True),
        )
        self.decoder = nn.Sequential(
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(128, 256),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(256, 512),
            nn.ReLU(True),
            nn.Dropout(0.5),
            nn.Linear(512, 28 * 28),
            nn.Sigmoid(),
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
