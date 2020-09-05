#!/usr/bin/env python3
#
# Various utility functions.
#
# Date: September 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import numpy as np

import torch

import generators


def get_samples(device, P, generator, generator_mean=None, generator_std=None, teacher=None,
                transformation=None, transformation_mean=None, transformation_std=None):
    """Generates a set of samples from the given generator and applies the given
    transformation.

    The outputs of the generator are centered and re-scaled, if the
    corresponding statistics are available. Same for the transformation.

    Parameters:
    -----------
    device : device on which to create the samples
    P : number of samples
    generator : generative model that transforms latent variables to inputs
    generator_mean : scalar
        mean element of the generator output vector
    generator_std : scalar
        standard dev of the generator output
    teacher : teacher network
    transformation : the transform to be applied to the generated inputs
    transformation_mean : scalar
        vector with mean of the transformation output
    transformation_std : scalar
        standard dev of the transformation output

    """
    with torch.no_grad():
        cs = torch.randn(P, generator.N_in).to(device)

        # propagate latent variables through the generator
        xs = generator.transform(cs)

        # if statistics are available, center generator outputs
        if generator_mean is not None:
            xs = (xs - generator_mean) / generator_std

        # apply transform, if given
        if transformation is not None:
            if generator_mean is None or generator_std is None :
                msg = "Don't apply a transformation to uncentered outputs!"
                raise NotImplementedError(msg)

            xs = transformation.transform(xs)

            # if statistics are available, center transform outputs
            if transformation_mean is not None:
                xs = (xs - transformation_mean) / transformation_std

        # if a teacher is given, generate labels
        ys = None if teacher is None else teacher(cs)

        return cs, xs, ys


def get_generator(name, device):
    """
    Returns a new instance of the generator with the given name.

    Paramters:
    ----------
    name : name of the generator, e.g. dcgan_cifar100_grey
    device : device on which to create the samples

    Raises:
    -------
    ValueError : if the given name is not recognised as a generative model.
    """
    if name == "dcgan_rand":
        generator = generators.dcgan_cifar10(device, random=True)
    elif name == "dcgan_cifar10":
        generator = generators.dcgan_cifar10(device)
    elif name == "dcgan_cifar100_grey":
        generator = generators.dcgan_cifar100_grey(device)
    elif name == "nvp_cifar10":
        generator = generators.nvp_cifar10(device)
    else:
        raise ValueError("Did not recognise the generator here, will exit now.")

    return generator


def get_scalar_mean_std(mean, cov):
    """
    Given a vector mean and covariance matrix of a distribution in N dimensions, returns
    the scalar mean and standard deviation of data drawn from that distribution.

    Parameters:
    -----------
    mean : (N)
    cov : (N, N)
    """
    N = cov.shape[0]
    mean_scalar = torch.mean(mean)
    var_scalar = torch.mean(cov[np.diag_indices(N)] + mean**2) - torch.mean(mean)**2

    return mean_scalar, torch.sqrt(var_scalar)
