#!/usr/bin/env python3
#
# Defines super classes for Generator and Transformation,
# and provides various implementations.
#
# Date: September 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

from abc import ABCMeta, abstractmethod

import math

import kymatio.torch

import torch


class Transformation(metaclass=ABCMeta):
    """
    A transformation is any high-dimensional function that transforms high-dim
    inputs to high-dim outputs. Such a transformation could be a generative model
    transforming i.i.d. Gaussian random variables into images, or the scattering
    transform pre-processing some images.
    """

    @property
    @abstractmethod
    def name(self):
        """
        The name of this generator
        """
        pass

    @property
    @abstractmethod
    def N_in(self):
        """
        Input dimension of this generator
        """
        pass

    @property
    @abstractmethod
    def N_out(self):
        """
        Output dimension of this generator.
        """
        pass

    @abstractmethod
    def transform(self, inputs):
        """
        The core of this transformation: takes the samples cs, and transform them.

        Parameters:
        -----------
        inputs : (P, N_in)
            input to the transform; e.g. randomness fed into the generator.

        Returns:
        --------
        outputs : (P, N_out)
            tensor with transformed samples
        """
        pass


class NoTransform(Transformation):
    """
    Dummy class that creates no transformation
    """

    N_in = None
    N_out = None

    def __init__(self, N_in):
        self.N_in = N_in
        self.N_out = N_in

    def name(self):
        return "notransform"

    def transform(self, xs):
        return xs


class dcgan_cifar10(Transformation):
    """
    DCGAN of Radford et al. for CIFAR10 with random or pre-trained weights.
    """

    N_in = 100
    N_out = 3 * 32 * 32

    def __init__(self, device, random=False):
        """
        Initialises the DCGAN for CIFAR10, with either pre-trained or random weights.

        Parameters:
        -----------
        device : pyTorch device indicator (cpu vs gpu)
        random : if True, load random weights
        """
        self.device = device
        self.random = random
        from models.dcgan.dcgan_cifar10 import Generator

        self.generator = Generator(ngpu=1)
        # load weights
        loadweightsfrom = "weights/dcgan_%s_weights.pth" % (
            "rand" if random else "cifar10"
        )
        self.generator.load_state_dict(torch.load(loadweightsfrom, map_location=device))
        self.generator.eval()
        self.generator.to(device)

    def name(self):
        return "dcgan_%s" % ("rand" if self.random else "cifar10")

    def transform(self, cs):
        with torch.no_grad():
            latent = cs.unsqueeze(-1).unsqueeze(-1)
            xs = self.generator(latent)
            xs = xs.reshape(-1, self.N_out)
        return xs


class nvp_cifar10(Transformation):
    """
    real NVP of Dinh et al. (2017) pre-trained on CIFAR10.
    """

    N_in = 3 * 32 * 32
    N_out = N_in

    def __init__(self, device):
        """
        Initialises the DCGAN for CIFAR10.

        Parameters:
        -----------
        device : pyTorch device indicator (cpu vs gpu)
        """
        self.device = device

        import realnvp, data_utils
        from data_utils import Hyperparameters

        self.flow = torch.load("weights/nvp_cifar10.model", map_location=device)
        self.generator = self.flow.g

    def name(self):
        return "nvp_cifar10"

    def transform(self, cs):
        with torch.no_grad():
            latent = cs.reshape(-1, 3, 32, 32)
            xs = self.generator(latent)
            xs = xs.reshape(-1, self.N_out)

        return xs


class dcgan_cifar100_grey(Transformation):
    """
    DCGAN of Radford et al. for greyscale CIFAR100 with pre-trained weights.
    """

    N_in = 100
    N_out = 1 * 32 * 32

    def __init__(self, device):
        """
        Initialises the DCGAN for greyscale CIFAR100.

        Parameters:
        -----------
        device : pyTorch device indicator (cpu vs gpu)
        """
        self.device = device
        from models.dcgan.dcgan_cifar100 import Generator

        self.generator = Generator(ngpu=1)
        # load weights
        loadweightsfrom = "weights/dcgan_cifar100_grey_weights.pth"
        self.generator.load_state_dict(torch.load(loadweightsfrom, map_location=device))
        self.generator.eval()
        self.generator.to(device)

    def name(self):
        return "dcgan_cifar100_grey"

    def transform(self, cs):
        with torch.no_grad():
            latent = cs.unsqueeze(-1).unsqueeze(-1)
            xs = self.generator(latent)
            xs = xs.reshape(-1, self.N_out)
        return xs


class RandomProjection(Transformation):
    """
    Random projection with Gaussian weights and sign activation function.

    """
    N_in = None
    N_out = None

    def __init__(self, device, N_in, N_out):
        self.N_in = N_in
        self.N_out = N_out

        weights_fname = "weights/%s_weights.pth" % self.name()
        try:
            self.weights = torch.load(weights_fname).to(device)
        except FileNotFoundError:
            raise ValueError("Did not find weights for this transform.")

    def name(self):
        return "rand_proj_gauss_sign_from%d_to%d" % (self.N_in, self.N_out)

    def transform(self, xs):
        with torch.no_grad():
            zs = torch.sign(xs @ self.weights)
        return zs


class Scattering2D(Transformation):
    """
    Scattering transform in two dimensions.

    """
    N_in = None
    N_out = None

    def __init__(self, device, N_in, N_out):
        self.N_in = N_in
        self.N_out = 81 * 8 * 8

        self.width = int(math.sqrt(N_in))
        self.scattering = kymatio.torch.Scattering2D(J=2, shape=(self.width,
                                                                 self.width))
        self.scattering.to(device)

    def name(self):
        return "scattering2D"

    def transform(self, xs):
        with torch.no_grad():
            zs = xs.reshape(-1, self.width, self.width)
            zs = self.scattering(zs)
        zs = torch.flatten(zs, 1, -1)
        return zs
