from .greenlearning.utils import DataProcessor
from .greenlearning.model import GreenNN
from .chebpy2 import Chebfun2, Chebpy2Preferences, Quasimatrix
import numpy as np
from abc import ABC
import os,sys
import pathlib

matlab_path = "/Applications/MATLAB_R2023a.app/bin/matlab"

class ChebGreen(ABC):
    def __init__(self,
                Theta           : np.array,
                generateData    : bool = True,
                script          : str = "generate_example",
                example         : str = None,
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
        
        # Set data path for the model:
        if generateData: 
            print(f"Generating dataset for example \'{example}\'")
            self.datapath = self.generateMatlabData(script, example)
        else:
            print(f"Loading dataset at {datapath}")
            assert datapath is not None, "No datapath specified!"
            self.datapath = datapath

        # Load or fit greenlearning models on the dataset and learn a chebfun2 on them:
        sys.stdout.flush()
        print("-------------------------------------------------------------------------------\n")
        print("Generating chebfun2 models:")
        self.generateChebfun2Models(example)


    def generateMatlabData(self, script, example):
        for theta in self.Theta:
            if pathlib.Path(f"datasets/{example}/{theta:.2f}.mat").is_file():
                print(f"Dataset for for Theta = {theta:.2f}. Skipping dataset generation.")
                continue
            examplematlab = "\'"+example+"\'"
            matlabcmd = f"{matlab_path} -nodisplay -nosplash -nodesktop -r \"{script}({examplematlab},{theta:.2f}); exit;\" | tail -n +11"
            with open("temp.sh", 'w') as f:
                f.write(matlabcmd)
                f.close()
            os.system(f"bash temp.sh")
            os.remove("temp.sh")
        return os.path.abspath(f"datasets/{example}")
    

    def generateChebfun2Models(self, example):
        model = GreenNN()
        self.G = {}
        for theta in self.Theta:
            
            GreenNNPath = "savedModels/" + example + f"/{theta:.2f}"
            
            if model.checkSavedModels(loadPath = GreenNNPath):          # Check for stored models
                print(f"Found saved model, Loading model for example \'{example}\' at Theta = {theta:.2f}")
                model.build(dimension = 1, loadPath = GreenNNPath)
            else:
                data = DataProcessor(self.datapath + f"/{theta:.2f}.mat")
                data.generateDataset(trainRatio = 0.95, batch_size = 256)
                model.build(dimension = 1, layerConfig = [50,50,50,50], activation = 'rational')
                print(f"Training greenlearning model for example \'{example}\' at Theta = {theta:.2f}")
                lossHistory = model.train(data, epochs = 10000)
                model.saveModels(f"savedModels/{example}/{theta:.2f}")
            
            print(f"Learning a chebfun2 model for example \'{example}\' at Theta = {theta:.2f}")
            self.G[theta] = (Chebfun2(model.evaluateG, domain = [0, 1, 0, 1], prefs = Chebpy2Preferences(), simplify = False))
            print(f"Chebfun2 model added for example \'{example}\' at Theta = {theta:.2f}\n")
        
        maxRank = np.min(np.array([self.G[theta].rank for theta in self.Theta]))

        for theta in self.Theta:
            self.G[theta].truncate(maxRank)

    def generateNewModel(self, theta):
        pass

def compute_interp_coeffs(models : np.array, targetParam: np.array) -> np.array:
    """Computes the interpolation coefficients (based on fitting Lagrange polynomials) for performing interpolation,
    when the parameteric space is 1D. Note that the function takes in parameters of any dimnesions

    --------------------------------------------------------------------------------------------------------------------
    Arguments:
        models: Set of models (EGF objects) which are used to generate an interoplated model at the parameter
            targetParam.
        targetParam: An array of size (n_{param_dimension} x 1) array which defines the parameters for the model which
            we want to interpolate.

    --------------------------------------------------------------------------------------------------------------------
    Returns:
        A numpy array of Interpolation coefficents index in the same way as in the set models.
    """
    assert(not models == True)

    if (models[0].params.shape[0]!= 1):
        raise RuntimeError("Lagrange Polynomial based interpolation requires the parameteric space to be 1D.")


    a = np.ones(len(models))
    if len(models[0].params) == 1:
        thetas = [model.params[0] for model in models]
        theta = targetParam[0] # Define the target parameter
        for i,t1 in enumerate(thetas):
            for j,t2 in enumerate(thetas):
                if i != j:
                    a[i] = a[i]*((theta - t2)/(t1 - t2))

    return a




            
            

            

            