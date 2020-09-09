#!/usr/bin/env python3
#
# Create the weights of the random transformation with Gaussian weights.
#
# Date: Sep 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import torch


def main():
    N_in = 1024
    N_outs = [N_in, 2 * N_in, 4 * N_in, 8 * N_in]

    for N_out in N_outs:
        weights = torch.randn(N_in, N_out)
        fname = "rand_proj_gauss_sign_from%d_to%d_weights.pth" % (N_in, N_out)
        torch.save(weights, fname)


if __name__ == "__main__":
    main()
