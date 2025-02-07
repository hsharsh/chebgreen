# ChebGreen

ChebGreen is a **Python** library for learning and interpolating Green's function for 1-Dimensional problems in a continuous sense. It builds on our initial working on [learning and interpolating Green's functions using a manifold interpolation technique](https://www.sciencedirect.com/science/article/pii/S0045782523000944). The main idea is to learn a Green's function using Rational Neural Networks ([Greenlearning](https://greenlearning.readthedocs.io/en/latest/)), use our **Python** implementation of [chebfun](https://www.chebfun.org/) to learn a *continuous* Singular Value Expansion (SVE) for the bivariate Green's function, and then interpolate SVE on a manifold of *Quasimatrices*. Here's a small schematic to outline the first part of the process:

![Schematic for learning a Green's function](assets/schematic.png)

We use [chebpy](https://github.com/chebpy/chebpy) as a starting point to implement a **Python** version of [chebfun](https://www.chebfun.org/). The necessary features of chebfun in 2-Dimensions have been implemented along with bug fixes for chebpy. The implementation for the Rational Neural Networks to learn Green's functions is done in **Pytorch**.

## Installation

##### Create a virtual environment:
```bash
# Create folder for the virtual environment.
$ mkdir -p ~/.venvs # 

# Create a new virtual environment for the chebgreen package.
$ python -m venv ~/.venvs/chebgreen

# Activate environment (every time you want to use the package).
$ source ~/.venvs/chebgreen/bin/activate
```

##### Install the package

```bash
$ cd chebgreen

# Install the package and its dependencies.
$ pip install . -r requirements.txt
```

The code uses **MATLAB** and the **MATLAB** library Chebfun to generate the datasets. Instructions for installation can be found here:
- https://www.mathworks.com/products/matlab.html
- https://www.chebfun.org/download/

## Usage

The package provides some **MATLAB** scripts in the ``scripts`` directory for data generation. One can also load a dataset generated from another simulation software or from experiments, the format for the datasets is specified in ``chebgreen/model.py``.

The examples of using the package to learn and interpolate Green's function from a given Partial Differential Equation is inside ``Jupyter`` notebooks in ``examples`` directory. It also provides important visualizations for the learned Green's function, and computes an empirical error.