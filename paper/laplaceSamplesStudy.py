from chebgreen.greenlearning.utils import DataProcessor
from chebgreen.greenlearning.model import *
from chebgreen.backend import plt
from chebgreen.utils import runCustomScript

from chebgreen.chebpy2 import Chebfun2, Chebpy2Preferences
from chebgreen.chebpy2.chebpy.core.settings import ChebPreferences
from chebgreen.chebpy2.chebpy import chebfun, Chebfun
import shutil, time, sys

def green(x,s):
    g = 0
    g = (x <= s) * (x * (1-s)) + (x > s) * (s * (1-x))
    return g

N_samples = [2, 4, 8, 16, 32, 64, 128, 256, 512]
noise_level = 0.0

example = 'laplace'
script = 'generate_example'
theta = None
lmbda = 0.005
Nf = 1000
Nu = 1000

dimension = 1
domain = [0,1,0,1]
layerConfig = [50,50,50,50]
activation = 'rational'
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
print("-------------------------------------------------------------------------------")

Error = []
for N_sample in N_samples:
    runCustomScript(script,example,theta,N_sample,lmbda,Nf,Nu,noise_level)
    data = DataProcessor(f"datasets/{example}/data.mat")
    data.generateDataset(trainRatio = 0.95)

    model = GreenNN()

    model.build(dimension, domain, layerConfig, activation, dirichletBC)

    print("-------------------------------------------------------------------------------")
    print(f"Training greenlearning model for example \'{example}\' with {N_sample} samples")
    lossHistory = model.train(data, epochs = {'adam':int(2000), 'lbfgs':int(0)})

    xF, xU = data.xF, data.xU
    x, y = np.meshgrid(xU, xF)
    G = model.evaluateG(x,y)
    N = model.evaluateN(data.xF)
    GreenNNPath = "temp/"
    model.build(dimension = dimension,domain = domain, dirichletBC = dirichletBC, loadPath = GreenNNPath, device = torch.device('cpu'))

    savePath = f"plots/{example}"
    Path(savePath).mkdir(parents=True, exist_ok=True)
    
    fig = plt.figure(figsize = (8,6))
    plt.contourf(x,y,G, 50, cmap = 'turbo', vmin = np.min(G), vmax = np.max(G))
    plt.colorbar()
    fig.savefig(f'{savePath}/{example}-{int(N_sample)}.png', dpi = fig.dpi)

    print("-------------------------------------------------------------------------------")
    print(f"Learning a chebfun model for example \'{example}\' with {N_sample} samples")
    # ChebGreen
    cheb2prefs = Chebpy2Preferences()
    Gcheb = Chebfun2(model.evaluateG, domain = model.domain, prefs = cheb2prefs, simplify = True)

    print("-------------------------------------------------------------------------------")
    print(f"Computing empirical error for example \'{example}\' with {N_sample} samples")
    
    # Compute the empirical error
    cheb2prefs = Chebpy2Preferences()
    cheb2prefs.prefx.eps = eps
    cheb2prefs.prefx.eps = eps
    e = Chebfun2(lambda x,y: Gcheb[x,y] - green(x,y), domain = model.domain, prefs = cheb2prefs, simplify = True)
    
    Error.append(e.norm()/gnorm)
    print(f"Error for a model with {N_sample} samples is {Error[-1]*100}%")
    print("-------------------------------------------------------------------------------")
    shutil.rmtree(f"datasets/{example}")
for error in Error:
    print(f"{error*100:.2f}")
t1 = time.perf_counter()