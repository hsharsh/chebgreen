from .greenlearning.utils import DataProcessor
from .greenlearning.model import GreenNN
import numpy as np
from abc import ABC
import os

matlab_path = "/Applications/MATLAB_R2022a.app/bin/matlab"

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
        print("-------------------------------------------------------------------------------\n")
        print("Generating/Loading greenlearning models:")
        self.generateChebfun2Models(example)


    def generateMatlabData(self, script, example):
        for theta in self.Theta:
            examplematlab = "\'"+example+"\'"
            cmd = f"{matlab_path} -nodisplay -nosplash -nodesktop -r \"{script}({examplematlab},{theta:.2f}); exit;\" | tail -n +11"
            os.system(cmd)
        return os.path.abspath(f"datasets/{example}")
    

    def generateChebfun2Models(self, example):
        
        for theta in self.Theta:
            model = GreenNN()

            GreenNNPath = "savedModels/" + example + f"/{theta:.2f}"
            
            if model.checkSavedModels(loadPath = GreenNNPath):          # Check for stored models
                model.build(dimension = 1, loadPath = GreenNNPath)
            else:
                data = DataProcessor(self.datapath + f"/{theta:.2f}.mat")
                data.generateDataset(valRatio = 0.95, batch_size = 256)
                model.build(dimension = 1, layerConfig = [50,50,50,50], activation = 'rational')
                print(f"Training greenlearning model for Example \'{example}\' at Theta = {theta:.2f}")
                lossHistory = model.train(data, epochs = 3000)
                model.saveModels(f"savedModels/{example}/{theta:.2f}")


            
            

            

            