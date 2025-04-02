#!../.venv/bin python3.12
 
from chebgreen.greenlearning.utils import DataProcessor
from chebgreen.greenlearning.model import *
from chebgreen.backend import plt
from chebgreen.utils import runCustomScript

from chebgreen.chebpy2 import Chebfun2, Chebpy2Preferences
from chebgreen.chebpy2.chebpy.core.settings import ChebPreferences
from chebgreen.chebpy2.chebpy import chebfun, Chebfun
from chebgreen import ChebGreen
import shutil, time, sys

def green(x,s):
    g = 0
    num1 = (np.exp(3*x/2) - np.exp(3/2)) * (np.exp(1.5*(1+s)) - 1) * np.heaviside(x-s, 0.5)
    num2 = (np.exp(3*s/2) - np.exp(3/2)) * (np.exp(1.5*(1+x)) - 1) * np.heaviside(s-x, 0.5)
    factor = (2 * np.exp(s/2 - 2*x)) / (3 * (np.exp(3) - 1))
    g = factor * (num1 + num2)
    return g

# noise_levels = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
noise_levels = [0, 0.1]


Theta = [1.0,2.0,3.0]
theta_ = 2.5
domain = [-1,1,-1,1]
generateData = True
script = "generate_example"
example = "advection_diffusion"
dirichletBC = True

t0 = time.perf_counter()

print(f"Learning a chebfun model for analytical Green's function of Laplacian operator")
# Analytical Green's function
eps = 1e-6
cheb2prefs = Chebpy2Preferences()
cheb2prefs.prefx.eps = eps
cheb2prefs.prefx.eps = eps
g = Chebfun2(green, domain = domain, prefs = cheb2prefs, simplify = True)
gnorm = g.norm()

Error = []
for noise_level in noise_levels:
    print("-------------------------------------------------------------------------------")
    print(f"Learning a chebfun model for example \'{example}\' with {noise_level*100}% noise")
    model = ChebGreen(Theta, domain, generateData, script, example, dirichletBC)

    print("-------------------------------------------------------------------------------")
    print(f"Interpolating a chebfun model for example \'{example}\' with {noise_level*100}% noise")
    Ginterp, Ninterp = model.generateNewModel(theta_)

    savePath = f"plots-interp/{example}"
    Path(savePath).mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize = (8,6))
    model.interpG[theta_].plot(fig = fig)
    fig.savefig(f'{savePath}/{example}-{int(noise_level*100)}.png', dpi = fig.dpi)

    print("-------------------------------------------------------------------------------")
    print(f"Computing empirical error for example \'{example}\' with {noise_level*100}% noise")
    # Compute the empirical error
    cheb2prefs = Chebpy2Preferences()
    cheb2prefs.prefx.eps = eps
    cheb2prefs.prefx.eps = eps
    e = Chebfun2(lambda x,y: np.abs(Ginterp[x,y] - green(x,y)), domain = model.domain, prefs = cheb2prefs, simplify = False)
    
    Error.append(e.norm()/gnorm)
    print(f"Error for a model with {noise_level*100} % noise is {Error[-1]*100}%")
    print("-------------------------------------------------------------------------------")
    shutil.rmtree(f"datasets/{example}")
    t1 = time.perf_counter()
    print(f"Time taken so far: {t1-t0:.2f} seconds")
for error in Error:
    print(f"{error*100:.2f}")