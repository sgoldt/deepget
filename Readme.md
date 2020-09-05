# Code to explore the deep GET

Here we provide the code used to study the deep GET. The code provides
abstractions to implement combinations of generative models + static
transformations, as well as code to train shallow neural networks on this data
in a hidden manifold setup. Work in progress!


There are several parts to this package: (for step-by-step explanations, see
below)

| File                          | Description                                                                                                                                                    |
|-------------------------------|----------------------------------------------------------------------------------------------------------------------------------------------------------------|
| ```covariance_generator.py``` | Estimates the covariances of a generative neural network,<br>implemented using the [pyTorch](http://pytorch.org/) library,                                     |
| ```dcgan.py```                | An implementation of a deep convolutional GAN of Radford et<br>al. [1], provided by  [pyTorch examples](https://github.com/pytorch/examples/tree/master/dcgan) |
| ```deepgen_online.py```       | Trains two-layer neural networks when inputs are drawn from a generator                                                                                        |
| ```generators.py```           | Abstractions to implements various generators and transformation                                                                                                      |
| ```models```                  | random and pre-trained weights used for the experiments,<br>as well as the corresponding covariance matrices.                                                  |
| ```twolayer.py```             | Python utility functions                                                                                                                                       |
| ```realnvp.py```              | pyTorch implementation of real NVP model by [Fangzhou Mu](https://github.com/fmu2)                                                                             |
| ```data_utils.py```           | Utility functions for real NVP model by [Fangzhou Mu](https://github.com/fmu2)                                                                                 |

## External packages included in this repository

We were fortunate to be able to use the implementation of the DCGAN from the
pyTorch example repository, provided together with pre-trained weights by
[Chandan Singh](https://github.com/csinva). We are also grateful to [Fangzhou
Mu](https://github.com/fmu2) for his pyTorch port of the original real NVP
implementation. We include both these packages in this repository to make
reproducing the paper's experiments as easy as possible, but you should check
out the other work of Chandan and Fangzhou, too !
