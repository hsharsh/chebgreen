from .backend import os, sys, Path, MATLABPath, parser, np
from chebGreen.chebpy2.chebpy.core.algorithms import chebpts2
from chebGreen.chebpy2.chebpy.api import chebfun

def runCustomScript(script      : str,
                    example     : str   = "data",
                    theta       : float  = None,
                    Nsample     : int   = parser['MATLAB'].getint('Nsample'),
                    lmbda       : int   = parser['MATLAB'].getfloat('lambda'),
                    Nf          : int   = parser['MATLAB'].getint('Nf'),
                    Nu          : int   = parser['MATLAB'].getint('Nu'),
                    noise_level : float = parser['MATLAB'].getfloat('noise')):
    """
    Arguments:
    script: Name of the matlab script to run
    example: Name of the example to run/Name for the save file
    Nsample: Number of sampled pairs f/u
    lambda: Lengthscale of kernel for sampling f
    Nf: Discretization for f, lambda > (size of domain)/Nf 
    Nu: Discretization for u
    noise_level: Noise level of the solutions u
    theta: Control parameter value for the problem

    Returns:
    Nothing. Saves the data in the datasets folder.
    """
    sys.stdout.flush() # Flush the stdout buffer

    # # Set the appropriate example name
    example = "\'"+example+"\'"

    # Depending on the script type, define the appropriate MATLAB command
    if theta is None:
        matlabcmd = " ".join(f"{MATLABPath} -nodisplay -nosplash -nodesktop -r \
        \"addpath('scripts');\
        {script}({example},{int(Nsample)},{lmbda},{int(Nf)},{int(Nu)},{noise_level:.2f}); exit;\" | tail -n +11".split())
    else:
        matlabcmd = " ".join(f"{MATLABPath} -nodisplay -nosplash -nodesktop -r \
        \"addpath('scripts');\
        {script}({example},{int(Nsample)},{lmbda},{int(Nf)},{int(Nu)},{noise_level:.2f},{theta:.2f}); exit;\" | tail -n +11".split())

    # Write the MATLAB command to a temporary file and run it
    with open("temp.sh", 'w') as f:
        f.write(matlabcmd)
        f.close()

    os.system(f"bash temp.sh") # Run the temporary file
    os.remove("temp.sh") # Remove the temporary file
    sys.stdout.flush() # Flush the stdout buffer

def generateMatlabData(script: str, example: str, Theta: list = None):
    if Theta is None:
        runCustomScript(script, example) # Run the custom script without theta
    else:
        for theta in Theta:
            # Check if the dataset already exists
            if Path(f"datasets/{example}/{theta:.2f}.mat").is_file():
                print(f"Dataset found for Theta = {theta:.2f}. Skipping dataset generation.")
                continue
            sys.stdout.flush() # Flush the stdout buffer
            runCustomScript(script, example, theta) # Run the custom script with theta

    sys.stdout.flush() # Flush the stdout buffer
    return os.path.abspath(f"datasets/{example}") # Return the path to the dataset

def vec2cheb(f, x):
    f = f.reshape((-1))
    x = x.reshape((-1))
    domain = [np.min(x), np.max(x)] # Compute the bounds of the domain
    
    # Function is resampled at twice as many sampled points to maintain accuracy in a different basis.
    N = 2 * f.shape[0] # Check if this is fine
    
    # Compute the Chebyshev nodes and scale them to the domain
    xc = (chebpts2(N) + 1) * ((domain[1] - domain[0])/2) + domain[0]

    # Compute the interpolated value of the function at the Chebyshev nodes
    fc = np.interp(xc, x, f).reshape((-1,1))

    return chebfun(fc, domain)

def computeEmpiricalError(G, data):
    RE, UC, U0 = [],[],[]
    for i in range(data.valDataset[1].numpy().shape[0]):
        xF, xU = data.xF, data.xU
        f, u  = data.valDataset[0].numpy()[i,:], data.valDataset[1].numpy()[i,:]
        f0, u0 = vec2cheb(f,xF), vec2cheb(u,xU)
        uc = G.integralTransform(f0) + chebfun(G.evaluateN, domain = f0.domain)
        re = (uc - u0).abs()/np.abs(u0.sum())
        RE.append(re)
    
    error = np.mean([re.sum() for re in RE])
    return error