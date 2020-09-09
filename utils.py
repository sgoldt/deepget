#!/usr/bin/env python3
#
# Various utility functions.
#
# Date: September 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import re

import numpy as np

import torch

import transformations


def get_samples(
    device,
    P,
    generator,
    generator_mean=None,
    teacher=None,
    transformation=None,
    transformation_mean=None,
):
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
            xs -= generator_mean

        # apply transform, if given
        if transformation is not None:
            if generator_mean is None:
                msg = "Don't apply a transformation to uncentered outputs!"
                raise NotImplementedError(msg)

            xs = transformation.transform(xs)

            # if statistics are available, center transform outputs
            if transformation_mean is not None:
                xs -= transformation_mean

        # if a teacher is given, generate labels
        ys = None if teacher is None else teacher(cs)

        return cs, xs, ys


def get_inputs_by_name(
    device, P, generator_name, transformation_name,
):
    """Generates a set of samples from the generator and transformation with the given names.

    The outputs of the generator are centered and re-scaled, if the
    corresponding statistics are available. Same for the transformation.

    Parameters:
    -----------
    device : device on which to create the samples
    P : number of samples
    generator_name : generative model that transforms latent variables to inputs
    transformation_name : the transform to be applied to the generated inputs
    """
    # Find the right generator for the given scenario
    generator = get_generator(generator_name, device)
    # transformation of the inputs
    transformation = get_transformation(transformation_name, generator, device)

    model_desc = generator.name()
    if transformation is not None:
        model_desc += "_" + transformation.name()

    # get the moments of the generator to center its outputs
    try:
        generator_mean_vec = torch.load(
            "moments/%s_mean_x.pt" % generator.name(), map_location=device
        )
        generator_cov = torch.load(
            "moments/%s_omega.pt" % generator.name(), map_location=device
        )
        print("Loaded moments of generator " + generator.name())
    except FileNotFoundError:
        raise ValueError("Could not find moments for generator %s!" % generator.name())

    # define the scalar moments of the generator's output distribution
    generator_mean, generator_std = get_scalar_mean_std(
        generator_mean_vec, generator_cov
    )

    # Now get the moments of the inputs that come out of the transformation
    transformation_mean = None  # scalar mean
    transformation_std = None  # scalar standard deviation
    Omega = None  # input - input covariance
    try:
        mean_x = torch.load("moments/%s_mean_x.pt" % model_desc, map_location=device,)
        Omega = torch.load("moments/%s_Omega.pt" % model_desc, map_location=device,)

        transformation_mean, transformation_std = get_scalar_mean_std(mean_x, Omega)
    except FileNotFoundError:
        pass

    cs, xs, _ = get_samples(
        device,
        P,
        generator,
        generator_mean,
        None,  # no teacher
        transformation,
        transformation_mean,
    )

    return cs, xs


def get_generator(name, device):
    """
    Returns a new instance of the generator with the given name.

    Parameters:
    ----------
    name : name of the generator, e.g. dcgan_cifar100_grey
    device : device on which to create the samples

    Raises:
    -------
    ValueError : if the given name is not recognised as a generative model.
    """
    if name == "dcgan_rand":
        generator = transformations.dcgan_cifar10(device, random=True)
    elif name == "dcgan_cifar10":
        generator = transformations.dcgan_cifar10(device)
    elif name == "dcgan_cifar100_grey":
        generator = transformations.dcgan_cifar100_grey(device)
    elif name == "nvp_cifar10":
        generator = transformations.nvp_cifar10(device)
    else:
        raise ValueError("Did not recognise the generator here, will exit now.")

    return generator


def get_transformation(name, generator, device):
    """
    Returns a new instance of the transformation with the given name.

    Parameters:
    ----------
    name : name of the generator, e.g. rand_proj_gauss_sign_to2048
    device : device on which to create the samples

    Raises:
    -------
    ValueError : if the given name is not recognised as a generative model.

    """
    transformation = None
    if name is None:
        transformation = None
    elif name.startswith("rand_proj_gauss_sign"):
        temp = re.findall(r"\d+", name)
        [N_in, N_out] = list(map(int, temp))
        # create the random transformation
        transformation = transformations.RandomProjection(device, N_in, N_out)
    else:
        raise ValueError("Did not recognise the transformation, will exit now.")

    return transformation


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
    var_scalar = torch.mean(cov[np.diag_indices(N)] + mean ** 2) - torch.mean(mean) ** 2

    return mean_scalar, torch.sqrt(var_scalar)
