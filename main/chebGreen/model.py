from .greenlearning.utils import DataProcessor
from .greenlearning.model import GreenNN
from .chebpy2 import Chebfun2, Chebpy2Preferences, Quasimatrix
from .chebpy2.chebpy import chebfun
from .backend import os, sys, Path, np, ABC, MATLABPath
from .utils import generateMatlabData

class ChebGreen(ABC):
    def __init__(self,
                Theta           : np.array,
                domain          : list = [0, 1, 0, 1],
                generateData    : bool = True,
                script          : str = "generate_example",
                example         : str = None,
                homogenousBC    : bool = True,
                datapath        : str = None
                ):
        super().__init__()
        """
        Arguments:        
        Theta: A numpy array of shape (N_models, N_dimension) which specifies parameteric value at
        which the models are specified or need to be evaluated at.

        generateData: If set to True, "path" should be the location of the matlab script which is
        run at the parametric values in Theta to generate the dataset.

        script: A string which specifies the name matlab script in the current directory for
        generating the dataset.

        example: This specifies the name of the example which the user wants to run.

        data: If generateData is set to False, then user must provide the path to the dataset, which
        should be in the following format:
        ---------------------------------{Specify Data Format here}---------------------------------
        """
        
        self.Theta = Theta

        if type(domain) is not np.ndarray and type(domain) is list:
            self.domain = np.array(domain)
        else:
            self.domain = domain
        
        self.homogenousBC = homogenousBC

        # Set data path for the model:
        if generateData: 
            print(f"Generating dataset for example \'{example}\'")
            self.datapath = generateMatlabData(script, example, self.Theta)
        else:
            print(f"Loading dataset at {datapath}")
            assert datapath is not None, "No datapath specified!"
            self.datapath = datapath

        # Load or fit greenlearning models on the dataset and learn a chebfun2 on them:
        sys.stdout.flush()
        print("-------------------------------------------------------------------------------\n")
        print("Generating chebfun2 models:")
        self.generateChebfun2Models(example)
        self.interpG = {}

    def generateChebfun2Models(self, example):
        model = GreenNN()
        self.G = {}
        for theta in self.Theta:
            
            GreenNNPath = "savedModels/" + example + f"/{theta:.2f}"
            
            if model.checkSavedModels(loadPath = GreenNNPath):          # Check for stored models
                print(f"Found saved model, Loading model for example \'{example}\' at Theta = {theta:.2f}")
                model.build(dimension = 1, homogeneousBC = self.homogenousBC, loadPath = GreenNNPath)
            else:
                data = DataProcessor(self.datapath + f"/{theta:.2f}.mat")
                data.generateDataset(trainRatio = 0.95)
                model.build(dimension = 1, domain = self.domain, layerConfig = [50,50,50,50], activation = 'rational', homogeneousBC = self.homogenousBC,)
                print(f"Training greenlearning model for example \'{example}\' at Theta = {theta:.2f}")
                lossHistory = model.train(data, epochs = {'adam':int(5000), 'lbfgs':int(0)})
                model.saveModels(f"savedModels/{example}/{theta:.2f}")
            
            print(f"Learning a chebfun2 model for example \'{example}\' at Theta = {theta:.2f}")
            self.G[float(theta)] = (Chebfun2(model.evaluateG, domain = self.domain, prefs = Chebpy2Preferences(), simplify = True))
            print(f"Chebfun2 model added for example \'{example}\' at Theta = {theta:.2f}\n")
        
        maxRank = np.min(np.array([self.G[theta].rank for theta in self.Theta]))

        for theta in self.Theta:
            self.G[theta].truncate(maxRank)

    def generateNewModel(self, theta):
        if theta not in list(self.interpG.keys()):
            self.interpG[theta] = modelInterp(self.G, theta)
        return self.interpG[theta]

def computeInterpCoeffs(interpParams : list, targetParam: float) -> np.array:
    """Computes the interpolation coefficients (based on fitting Lagrange polynomials) for performing interpolation,
    when the parameteric space is 1D. Note that the function takes in parameters of any dimnesions

    --------------------------------------------------------------------------------------------------------------------
    Arguments:
        interpSet: Dictionary of models (Chebfun2 objects) which are used to generate an interoplated model at the
            parameter targetParam.
        targetParam: A float which defines the parameter at which we want to find a new model.

    --------------------------------------------------------------------------------------------------------------------
    Returns:
        A dict of Interpolation coefficents index
    """
    
    assert len(interpParams) > 2, "Need at least two interpolant models"
    assert all([isinstance(theta,float) for theta in interpParams]), \
        "Lagrange Polynomial based interpolation requires the parameteric space to be 1D."

    interpCoeffs = dict([(theta, 1.0) for theta in interpParams])

    for i,t1 in enumerate(interpCoeffs):
        for j,t2 in enumerate(interpCoeffs):
            if i != j:
                interpCoeffs[t1] = interpCoeffs[t1]*((targetParam - t2)/(t1 - t2))

    return interpCoeffs

def computeOrderSigns(R0: Quasimatrix, R1: Quasimatrix) -> tuple([np.array, np.array]):
    """ Given two orthonormal matrices R0 and R1, this function computes the "correct" ordering and signs of the columns
    (modes) of R1 using R0 as a reference. The assumption is that these are orthonormal matrices, the columns of which
    are eigenmodes of systems which are close to each other and hence the eigenmodes will close to each other as well.
    We thus find an order such that the modes of the second matrix have the maximum inner product (in magnitude) with
    the corresponding mode from the first matrix. If such an ordering doesn't exist the function raises a runtime error.

    Once such an ordering is found, one can flip the signs for the modes of R1, if the inner product is not positive.
    This is necessary when we want to interpolate.

    --------------------------------------------------------------------------------------------------------------------
    Args:
        R0: Orthonormal quasimatrix, the columns of which are used as the reference to re-order and find signs
        R1: Orthonormal quasimatrix for which the columns are supposed to be reordered.

    --------------------------------------------------------------------------------------------------------------------
    Returns:
        New ordering and the signs (sign flips) for matrix R1.
    """
    rank = R0.shape[1]
    order = -1*np.ones(rank).astype(int)
    signs = np.ones(rank)
    
    used = set()
    # For each mode in R1, Search over all modes of R0 for the best matching mode.
    products = np.abs(R0.T * R1) # Compute all the pairwise innerproducts
    for i in range(rank):
        maxidx, maxval = -1, -1
        for j in range(rank):
            current = products[i,j]
            if current >= maxval and (j not in used):
                maxidx = j
                maxval = current
        order[i] = maxidx
        used.add(maxidx)
    
    # Raise an error if the ordering of modes is not a permutation.
    check = set()
    for i in range(rank):
        check.add(order[i])

    if len(check) != rank:
        raise RuntimeError('No valid ordering of modes found')
    
    # Signs are determined according to the correct ordering of modes
    for i in range(rank):
        if (R0[:,i].T * R1[:,int(order[i])]).item() < 0:
            signs[i] = -1
    
    return order, signs

def modelInterp(interpSet: dict[float,Chebfun2], targetParam: float) -> Chebfun2:
    """
    Interpolation for the models. The left and right singular functions are interpolated in the tangent space of
    (L^2(domain))^K (K is the model rank) using a QR based retraction map. The singular values are interpolated
    directly (entry-by-entry) using a Lagrange polynomial based inteporlation. Note that currently the interpolation
    only supports 1D parameteric spaces for the model as the method for interpolation within the tangent space only
    supports 1D parameteric spaces but this can be easily extended to higher dimensions. The lifting and retraction to
    the tangent space at an "origin" has no dependendence on the dimensionality of the parameteric space.

    --------------------------------------------------------------------------------------------------------------------
    Arguments:
        interpSet: Dictionary of models (Chebfun2 objects) which are used to generate an interoplated model at the
            parameter targetParam.
        targetParam: A float which defines the parameter at which we want to find a new model.

    --------------------------------------------------------------------------------------------------------------------
    Returns:
        An Chebfun2 object at the target parameter.
    """
    interpParams = list(interpSet.keys())

    assert len(interpParams) > 2, "Need at least two interpolant models"

    # Find the model which is closest to target parameter. Note that the distance is calculated in terms of norm of the
    # normalized parameters.
    refIndex = None
    minDistance = np.inf
    for theta in interpParams:
        distance = np.linalg.norm((theta-targetParam)/targetParam)
        if distance < minDistance:
            minDistance = distance
            refIndex = theta
    
    interpCoeffs = computeInterpCoeffs(interpParams, targetParam)
    
    # Define the origin
    U0, _, Vt0 = interpSet[refIndex].cdr()
    V0 = Vt0.T
    K = interpSet[refIndex].rank
    
    U_ = Quasimatrix(data = chebfun(np.zeros((2,K)), domain = interpSet[refIndex].domain[2:]), transposed = False)
    S_ = np.zeros(K)
    V_ = Quasimatrix(data = chebfun(np.zeros((2,K)), domain = interpSet[refIndex].domain[:2]), transposed = False)
    
    for theta, model in interpSet.items():
        U, S, Vt = model.cdr()
        order, signs = computeOrderSigns(U0,U)
        
        Uc = U[:,order] * np.diag(signs)
        S = np.diag(S)[order]
        Vc = (Vt.T)[:,order] * np.diag(signs)
        
        # Project to tangent space of model at origin
        Up = Uc - U0 * ((U0.T * Uc + Uc.T * U0)*0.5)
        Vp = Vc - V0 * ((V0.T * Vc + Vc.T * V0)*0.5)
        
        
        # Interpolate the singular functions
        U_ += Up * np.diag(np.ones(K)*interpCoeffs[theta])
        V_ += Vp * np.diag(np.ones(K)*interpCoeffs[theta])
        
        # Interpolate the singular values directly
        S_ += S * interpCoeffs[theta]
        
        
    Un, _ = (U0 + U_).qr()
    Vn, _ = (V0 + V_).qr()
    
    # Match the order and signs with the origin
    order, signs = computeOrderSigns(U0,Uc)
    Un = Un[:,order] * np.diag(signs)
    Sn = S_[order]
    Vtn = (Vn[:,order] * np.diag(signs)).T
    
    
    return Chebfun2([Un, Sn, Vtn])




            
            

            

            