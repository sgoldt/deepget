#!/usr/bin/env python3
#
# Robust estimation of the mean and covariance of an arbitray generator,
# potentially after applying a non-trivial transformation.
#
# Date: Sep 2020 (v2)
#       May 2020 (v1)
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import argparse

import utils

import torch
from tqdm import tqdm


def log(msg, logfile):
    """
    Print log message to  stdout and the given logfile.
    """
    print(msg)
    logfile.write(msg + "\n")


def main():
    parser = argparse.ArgumentParser()
    device_help = "which device to run on: 'cuda:x' or 'cpu'"
    generator_help = "Generator of the inputs: dcgan_rand, dcgan_cifar10, dcgan_cifar100_grey, nvp_cifar10."
    transform_help = "Transform: identity, ..."
    checkpoint_help = "checkpoint every ... steps"
    seed_help = "random number generator seed."
    parser.add_argument("--generator", help=generator_help, default="rand")
    parser.add_argument("--transform", help=transform_help, default="identity")
    parser.add_argument("--device", "-d", help=device_help)
    parser.add_argument("--bs", type=int, default=4096, help="batch size.")
    parser.add_argument("--steps", type=int, default=1e9, help="number of steps")
    parser.add_argument("--checkpoint", type=int, default=1000, help=checkpoint_help)
    parser.add_argument("-q", "--quiet", help="be quiet", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0, help=seed_help)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Will use chunks of data of size (batch_size, N) or (batch_size, D) etc.
    batch_size = args.bs

    # Find the right generator...
    generator = utils.get_generator(args.generator, device)
    # ... and transformation of the inputs
    transformation = utils.get_transformation(args.transform, generator, device)

    # Define the dimensions of the problem
    D = generator.N_in
    N = generator.N_out if transformation is None else transformation.N_out
    # and its moments
    generator_mean = None
    generator_std = None

    # If we want to estimate the moments of generator + transform, load moments
    # of the generator first
    if transformation is not None:
        try:
            generator_mean_vec = torch.load(
                "moments/%s_mean_x.pt" % generator.name(), map_location=device
            )
            generator_cov = torch.load(
                "moments/%s_omega.pt" % generator.name(), map_location=device
            )
            generator_mean, generator_std = utils.get_scalar_mean_std(
                generator_mean_vec, generator_cov
            )
        except FileNotFoundError:
            print(
                "Could not find moments of generator. Can therefore not estimate "
                "moments of generator + transformation. Will exit now!"
            )
            exit()

    max_P = args.steps * args.bs
    transform_name = "" if transformation is None else transformation.name() + "_"
    log_fname = "covariance_%s_%sP%g_s%d.dat" % (
        generator.name(),
        transform_name,
        max_P,
        args.seed,
    )
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Computing the covariance for %s" % generator.name()
    if transformation is None:
        welcome += "\n"
    else:
        welcome += " with transformation %s\n" % transformation.name()
    welcome += "# batch size=%d, seed=%d\n" % (batch_size, args.seed)
    welcome += "# Using device: %s\n" % str(device)
    welcome += "# samples, diff E c, diff E x, diff Omega, diff Phi"
    log(welcome, logfile)

    # Hold the Monte Carlo estimators computed here
    variables = ["mean_c", "mean_x", "omega", "phi"]
    mc = {
        "mean_c": torch.zeros(D).to(device),  # estimate of mean of c
        "mean_x": torch.zeros(N).to(device),  # estimate of mean of x
        "omega": torch.zeros(N, N).to(device),  # input-input covariance
        "phi": torch.zeros(N, D).to(device),  # input-latent covariance
    }
    M2_omega = torch.zeros(N, N).to(device)  # running estimate of residuals
    M2_phi = torch.zeros(N, D).to(device)  # running estimate of residuals

    # store the values of the covariance matrices at the last checkpoint
    mc_last = dict()
    for name in variables:
        mc_last[name] = torch.zeros(mc[name].shape).to(device)

    step = -1
    with torch.no_grad():
        while step < args.steps:
            for _ in tqdm(range(args.checkpoint)):
                # slighly unsual place for step increment; is to preserve the usual notation
                # when computing the current estimate of the covariance outside this loop
                step += 1

                # Generate a new batch of data
                cs, xs, _ = utils.get_samples(
                    device,
                    batch_size,
                    generator,
                    generator_mean,
                    generator_std,
                    transformation=transformation,
                )

                # Update the estimators.
                ########################
                mc_mean_x_old = mc["mean_x"]
                # Start with the means
                dmean_c = torch.mean(cs, axis=0) - mc["mean_c"]
                mc["mean_c"] += dmean_c / (step + 1)
                dmean_x = torch.mean(xs, axis=0) - mc["mean_x"]
                mc["mean_x"] += dmean_x / (step + 1)
                # now the residuals
                M2_omega += (xs - mc_mean_x_old).T @ (xs - mc["mean_x"]) / batch_size
                M2_phi += (xs - mc_mean_x_old).T @ (cs - mc["mean_c"]) / batch_size

            mc["omega"] = M2_omega / (step + 1)
            mc["phi"] = M2_phi / (step + 1)

            # Build status message
            status = "%g" % (step * args.bs)
            for name in variables:
                diff = torch.sqrt(torch.mean((mc[name] - mc_last[name]) ** 2))
                status += ", %g" % diff

            log(status, logfile)

            # Write the estimates to files
            for name in variables:
                fname = log_fname[:-4] + ("_%s_%g.pt" % (name, step * batch_size))
                torch.save(mc[name], fname)

            for name in variables:
                mc_last[name] = mc[name].clone().detach()

        # Write the estimates to files
        for name in variables:
            fname = log_fname[:-4] + ("_%s_%g.pt" % (name, step * batch_size))
            torch.save(mc[name], fname)


if __name__ == "__main__":
    main()
