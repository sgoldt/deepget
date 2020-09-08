#!/usr/bin/env python3
#
# Training two-layer networks on inputs coming from various deep generators with
# static transformations applied to the inputs.
#
# Date: September 2020
#
# Author: Sebastian Goldt <goldt.sebastian@gmail.com>

import argparse
import math

import numpy as np  # for storing tensors in CSV format

import torch
import torch.optim as optim
import torch.nn as nn
import torch.nn.functional as F

from twolayer import TwoLayer, identity, erfscaled

import utils

NUM_TESTSAMPLES = 10000


class HalfMSELoss(nn.MSELoss):
    def __init__(self, size_average=None, reduce=None, reduction="mean"):
        super().__init__(size_average, reduce, reduction)

    def forward(self, input, target):
        return 0.5 * F.mse_loss(input, target, reduction=self.reduction)


def eval_student(time, student, test_xs, test_ys, nus, T, tildeT, A, criterion):
    N = test_xs.shape[1]

    student.eval()
    with torch.no_grad():
        # compute the generalisation error w.r.t. the noiseless teacher
        preds = student(test_xs)
        eg = criterion(preds, test_ys)

        w = student.fc1.weight.data
        v = student.fc2.weight.data
        lambdas = w.mm(test_xs.T) / math.sqrt(N)
        Q_num = lambdas.mm(lambdas.T) / NUM_TESTSAMPLES
        R_num = lambdas.mm(nus.T) / NUM_TESTSAMPLES

        eg_analytical = get_eg_analytical(Q_num, R_num, T, A, v)

        msg = "%g, %g, %g, nan, " % (time, eg, eg_analytical)

        # upper triangular elements for symmetric matrices
        indices_K = Q_num.triu().nonzero().T
        indices_M = T.triu().nonzero().T
        Q_num_vals = Q_num[indices_K[0], indices_K[1]].cpu().numpy()
        msg += ", ".join(map(str, Q_num_vals)) + ", "
        msg += ", ".join(map(str, R_num.flatten().cpu().numpy())) + ", "
        T_vals = T[indices_M[0], indices_M[1]].cpu().numpy()
        msg += ", ".join(map(str, T_vals)) + ", "
        tildeT_vals = tildeT[indices_M[0], indices_M[1]].cpu().numpy()
        msg += ", ".join(map(str, tildeT_vals)) + ", "
        msg += ", ".join(map(str, A.flatten().cpu().numpy())) + ", "
        msg += ", ".join(map(str, v.flatten().cpu().numpy())) + ", "

        return msg[:-2]


def get_eg_analytical(Q, R, T, A, v):
    """
    Computes the analytical expression for the generalisation error of erf teacher
    and student with the given order parameters.

    Parameters:
    -----------
    Q: student-student overlap
    R: teacher-student overlap
    T: teacher-teacher overlap
    A: teacher second layer weights
    v: student second layer weights
    """
    eg_analytical = 0
    # student-student overlaps
    sqrtQ = torch.sqrt(1 + Q.diag())
    norm = torch.ger(sqrtQ, sqrtQ)
    eg_analytical += torch.sum((v.t() @ v) * torch.asin(Q / norm))
    # teacher-teacher overlaps
    sqrtT = torch.sqrt(1 + T.diag())
    norm = torch.ger(sqrtT, sqrtT)
    eg_analytical += torch.sum((A.t() @ A) * torch.asin(T / norm))
    # student-teacher overlaps
    norm = torch.ger(sqrtQ, sqrtT)
    eg_analytical -= 2.0 * torch.sum((v.t() @ A) * torch.asin(R / norm))
    return eg_analytical / math.pi


def write_density(fname, density):
    """
    Stores the given order parameter density in a file of name fname in the Armadillo
    text format.

    Parameters:
    -----------
    density: (K, M, N)
    """
    K, M, N = density.shape
    output = open(fname, "w")
    output.write("ARMA_CUB_TXT_FN008\n")
    output.write("%d %d %d\n" % (K, M, N))
    for i in range(N):
        for k in range(K):
            for m in range(M):
                output.write("  %+.6e" % density[k, m, i])
        output.write("\n")

    output.close()


def log(msg, logfile):
    """
    Print log message to  stdout and the given logfile.
    """
    print(msg)
    logfile.write(msg + "\n")


def main():
    # define the command line arguments
    g_help = "teacher + student activation function: 'erf' or 'relu'"
    M_help = "number of teacher hidden nodes"
    K_help = "number of student hidden nodes"
    device_help = "which device to run on: 'cuda' or 'cpu'"
    generator_help = "Generator of the inputs: dcgan_rand, dcgan_cifar10, dcgan_cifar100_grey, nvp_cifar10."
    transform_help = "Transform: identity, scattering, ..."
    steps_help = "training steps as multiples of N"
    seed_help = "random number generator seed."
    parser = argparse.ArgumentParser()
    parser.add_argument("-g", "--g", default="erf", help=g_help)
    parser.add_argument("-M", "--M", type=int, default=2, help=M_help)
    parser.add_argument("-K", "--K", type=int, default=2, help=K_help)
    parser.add_argument("--generator", help=generator_help, default="rand")
    parser.add_argument("--transform", help=transform_help, default="identity")
    parser.add_argument("--device", "-d", help=device_help)
    parser.add_argument("--lr", type=float, default=0.2, help="learning rate")
    parser.add_argument("--bs", type=int, default=1, help="mini-batch size")
    parser.add_argument("--steps", type=int, default=10000, help=steps_help)
    parser.add_argument("-q", "--quiet", help="be quiet", action="store_true")
    parser.add_argument("-s", "--seed", type=int, default=0, help=seed_help)
    parser.add_argument("--store", action="store_true", help="store initial conditions")
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    if args.device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    (M, K, lr) = (args.M, args.K, args.lr)

    # Find the right generator for the given scenario
    generator = utils.get_generator(args.generator, device)
    # and define its scalar moments, which will be computed later
    generator_mean = None
    generator_std = None

    # transformation of the inputs
    transformation = utils.get_transformation(args.transform, generator, device)

    # Define the dimensions of the problem
    D = generator.N_in
    N = generator.N_out if transformation is None else transformation.N_out

    # output file + welcome message
    model_desc = generator.name()
    if transformation is not None:
        model_desc += "_" + transformation.name()
    log_fname = "transform_online_%s_D%d_N%d_%s_M%d_K%d_lr%g_i2_s%d.dat" % (
        model_desc,
        D,
        N,
        args.g,
        M,
        K,
        lr,
        args.seed,
    )
    logfile = open(log_fname, "w", buffering=1)
    welcome = "# Two-layer nets on inputs from generator %s" % generator.name()
    if transformation is None:
        welcome += "\n"
    else:
        welcome += " with transformation %s\n" % transformation.name()
    welcome += "# M=%d, K=%d, lr=%g, batch size=%d, seed=%d\n" % (
        M,
        K,
        lr,
        args.bs,
        args.seed,
    )
    welcome += "# Using device:" + str(device)
    log(welcome, logfile)

    # networks and loss
    g = erfscaled if args.g == "erf" else F.relu
    gs = (g, identity)
    student = TwoLayer(gs, N, args.K, 1, normalise1=True, std0=1e-2)
    student.to(device)

    teacher = TwoLayer(gs, D, args.M, 1, normalise1=True, std0=1)
    nn.init.constant_(teacher.fc2.weight, 1)
    teacher.freeze()
    teacher.to(device)
    B = teacher.fc1.weight.data
    A = teacher.fc2.weight.data

    # collect the parameters that are going to be optimised by SGD
    params = []
    params += [{"params": student.fc1.parameters()}]
    # If we train the last layer, ensure its learning rate scales correctly
    params += [{"params": student.fc2.parameters(), "lr": lr / N}]
    optimizer = optim.SGD(params, lr=lr)
    criterion = HalfMSELoss()

    print("# Generator, Teacher and Student: ")
    for net in [generator, teacher, student]:
        msg = "# " + str(net).replace("\n", "\n# ")
        log(msg, logfile)

    # when to print?
    end = torch.log10(torch.tensor([1.0 * args.steps])).item()
    times_to_print = list(torch.logspace(-1, end, steps=200))

    # get the moments of the generator to center its outputs
    try:
        generator_mean_vec = torch.load(
            "moments/%s_mean_x.pt" % generator.name(), map_location=device
        )
        generator_cov = torch.load(
            "moments/%s_omega.pt" % generator.name(), map_location=device
        )
    except FileNotFoundError:
        print(
            "Could not find moments of generator %s. Will exit now!" % generator.name()
        )
        exit()
    generator_mean, generator_std = utils.get_scalar_mean_std(
        generator_mean_vec, generator_cov
    )

    # generate the test set
    test_cs, test_xs, test_ys = utils.get_samples(
        device,
        NUM_TESTSAMPLES,
        generator,
        generator_mean,
        generator_std,
        teacher,
        transformation,
    )

    Omega = None  # the student input - input covariance
    Phi = None  # the generator input - student input covariance
    # Either load pre-computed Omega and Phi, or generate from the test set
    try:
        Omega = torch.load("moments/%s_Omega.pt" % model_desc, map_location=device,)
        Phi = torch.load("moments/%s_phi.pt" % model_desc, map_location=device,)
    except FileNotFoundError:
        Omega = 1 / NUM_TESTSAMPLES * test_xs.T @ test_xs
        Phi = 1 / NUM_TESTSAMPLES * test_xs.T @ test_cs

    nus = B.mm(test_cs.T) / math.sqrt(D)

    msg = "# test xs: mean=%g, std=%g; test ys: std=%g" % (
        torch.mean(test_xs),
        torch.std(test_xs),
        torch.std(test_ys),
    )
    log(msg, logfile)

    T = 1.0 / B.shape[1] * B @ B.T
    rotation = Phi.T @ Phi
    tildeT = 1 / N * B @ rotation @ B.T
    if args.store:
        with torch.no_grad():
            # compute the exact densities of r and q
            exq = torch.zeros((K, K, N), device=device)
            exr = torch.zeros((K, M, N), device=device)
            extildet = torch.zeros((M, M, N), device=device)
            sqrtN = math.sqrt(N)
            w = student.fc1.weight.data
            v = student.fc2.weight.data

            rhos, psis = torch.symeig(Omega, eigenvectors=True)
            rhos.to(device)
            psis.to(device)
            #  make sure to normalise, orient evectors according to the note
            psis = sqrtN * psis.T

            GammaB = 1.0 / sqrtN * B @ Phi.T @ psis.T
            GammaW = 1.0 / sqrtN * w @ psis.T

            for k in range(K):
                for l in range(K):
                    exq[k, l] = GammaW[k, :] * GammaW[l, :]
                for n in range(M):
                    exr[k, n] = GammaW[k, :] * GammaB[n, :]
            for n in range(M):
                for m in range(M):
                    extildet[n, m] = GammaB[n, :] * GammaB[m, :]

            root_name = log_fname[:-4]
            np.savetxt(root_name + "_T.dat", T.cpu().numpy(), delimiter=",")
            np.savetxt(root_name + "_rhos.dat", rhos.cpu().numpy(), delimiter=",")
            np.savetxt(root_name + "_T.dat", T.cpu().numpy(), delimiter=",")
            np.savetxt(root_name + "_A.dat", A[0].cpu().numpy(), delimiter=",")
            np.savetxt(root_name + "_v0.dat", v[0].cpu().numpy(), delimiter=",")

            write_density(root_name + "_q0.dat", exq)
            write_density(root_name + "_r0.dat", exr)
            write_density(root_name + "_tildet.dat", extildet)

    time = 0
    dt = 1 / N

    msg = eval_student(time, student, test_xs, test_ys, nus, T, tildeT, A, criterion)
    log(msg, logfile)
    while len(times_to_print) > 0:
        # get the inputs
        cs, inputs, targets = utils.get_samples(
            device,
            args.bs,
            generator,
            generator_mean,
            generator_std,
            teacher,
            transformation,
        )

        for i in range(args.bs):
            student.train()
            preds = student(inputs[i])
            loss = criterion(preds, targets[i])

            # TRAINING
            student.zero_grad()
            loss.backward()
            optimizer.step()

            time += dt

            if time >= times_to_print[0].item() or time == 0:
                msg = eval_student(
                    time, student, test_xs, test_ys, nus, T, tildeT, A, criterion
                )
                log(msg, logfile)
                times_to_print.pop(0)

    print("Bye-bye")


if __name__ == "__main__":
    main()
