from .backend import torch, np, Path, ABC, config, device
from .activations import get_activation
from .loss import LossGreensFunction



class NN(torch.nn.Module):
    def __init__(self,
                 numInputs = 2,
                 numOutputs = 1,
                 layerConfig = [50, 50, 50, 50],
                 activation = "rational",
                 dtype = config(torch),
                 device = device):
        # Initialize the Layers. We hold all layers in a ModuleList.

        super(NN, self).__init__()

        self.layers = torch.nn.ModuleList()
        self.activationFunctions = torch.nn.ModuleList()

        self.layers.append(torch.nn.Linear(
                                in_features  = numInputs,
                                out_features = layerConfig[0],
                                bias         = True ).to(dtype = dtype, device = device))
        self.activationFunctions.append(get_activation(activation))

        for neuronsIn, neuronsOut in zip(layerConfig[:-1],layerConfig[1:]):
            self.layers.append(torch.nn.Linear(
                                in_features  = neuronsIn,
                                out_features = neuronsOut,
                                bias         = True ).to(dtype = dtype, device = device))
            self.activationFunctions.append(get_activation(activation))

        self.layers.append(torch.nn.Linear(
                                in_features  = layerConfig[-1],
                                out_features = numOutputs,
                                bias         = True ).to(dtype = dtype, device = device))
        
        # Initialize layers
        for layer in self.layers:
            torch.nn.init.xavier_normal_(layer.weight)
            torch.nn.init.zeros_(layer.bias)

    def forward(self, X: torch.Tensor) -> torch.Tensor:

        # Loop over all the layers and activation functions.
        for layer, activation in zip(self.layers[:-1],self.activationFunctions):
            X = activation(layer(X))
        
        # Last layer does not take an activation function.
        return self.layers[-1](X)
    
class GreenNN(ABC):
    def __init__(self) -> None:
        super().__init__()

    def build(self, dimension = 1, layerConfig = [50, 50, 50, 50], activation = 'rational', loadPath = None, device = device):
        self.dimension = dimension
        self.device = device
        self.layerConfig = layerConfig
        self.activation = activation

        if loadPath == None:
            self.G = NN(numInputs = dimension*2, numOutputs = dimension, layerConfig = layerConfig, activation = activation).to(self.device)
            self.N = NN(numInputs = dimension, numOutputs = dimension, layerConfig = layerConfig, activation = activation).to(self.device)
        else:
            assert self.checkSavedModels(loadPath), "Saved models not found" 
            self.loadModels(loadPath, device = device)

    def train(self, data, epochs = {'adam':int(1000), 'lbfgs':int(200)}):
        
        params = list(self.G.parameters()) + list(self.N.parameters())

        self.optimizerAdam = torch.optim.Adam(params, lr = 1e-2)
        self.schedulerAdam = torch.optim.lr_scheduler.StepLR(self.optimizerAdam, step_size = 100, gamma = 0.9)
        self.optimizerLBFGS = torch.optim.LBFGS(params, lr = 1e-2)

        assert data.xF.shape[1] == self.dimension and data.xU.shape[1] == self.dimension,\
                f"Dimension of evaluation points for forcing, {data.xF.shape[1]}, and response, \
                {data.xU.shape[1]}, should match with the model dimension, {self.dimension}"
        
        self.init_loss(data.xF, data.xU)

        lossHistory = {'training': [], 'validation':[]}

        print("Training with Adam:")
        for epoch in range(epochs['adam']):
            for fTrain, uTrain in data.trainDataset:
                fTrain, uTrain = fTrain.to(device), uTrain.to(device)
                self.G.train()
                self.N.train()

                self.optimizerAdam.zero_grad()
                lossValue = self.lossfn(fTrain, uTrain)
                lossValue.backward()

                self.optimizerAdam.step()
                self.schedulerAdam.step()

            # Change this to be an average over batches. Currently assuming a single batch
            lossHistory['training'].append(lossValue.item())

            with torch.no_grad():
                for fVal, uVal in data.valDataset:
                    lossValue = self.lossfn(fVal.to(device), uVal.to(device))
            lossHistory['validation'].append(lossValue.item())
            if (epoch+1) % 100 == 0:
                print(f"Loss at epoch {epoch+1}: Training = {lossHistory['training'][-1]:.3E}, Validation = {lossHistory['validation'][-1]:.3E}")
        
        print("Training with LBFGS:")
        for epoch in range(epochs['lbfgs']):
            for fTrain, uTrain in data.trainDataset:
                fTrain, uTrain = fTrain.to(device), uTrain.to(device)
                self.G.train()
                self.N.train()

                def closure():
                    self.optimizerLBFGS.zero_grad()
                    lossValue = self.lossfn(fTrain, uTrain)
                    lossValue.backward()
                    return lossValue
                with torch.no_grad():
                    lossValue = self.lossfn(fTrain, uTrain)
                self.optimizerLBFGS.step(closure)

            
            # Change this to be an average over batches. Currently assuming a single batch
            lossHistory['training'].append(lossValue.item())

            with torch.no_grad():
                for fVal, uVal in data.valDataset:
                    lossValue = self.lossfn(fVal.to(device), uVal.to(device))
            lossHistory['validation'].append(lossValue.item())
            if (epoch+1) % 10 == 0:
                print(f"Loss at epoch {epoch+1}: Training = {lossHistory['training'][-1]:.3E}, Validation = {lossHistory['validation'][-1]:.3E}")

        return lossHistory
    
    # Not implemented for Green's function of dimension > 1
    def evaluateG(self, x, s):
        if isinstance(x,config(np)):
            x = np.array([x])
        if isinstance(s,config(np)):
            s = np.array([s])
        if isinstance(x.dtype, np.float64) or (s.dtype, np.float64):
            x, s = x.astype(config(np)), s.astype(config(np))
        assert(x.shape == s.shape), "Both input need to have the same shape"
        shape = x.shape
        if x.dtype == config(np):
            X = torch.tensor(np.vstack([x.ravel(), s.ravel()]).T, dtype = config(torch)).to(self.device)
        with torch.no_grad():
            G = self.G(X).cpu().numpy().reshape(shape)
        return G 


    def init_loss(self, xF, xU):
        self.lossfn = LossGreensFunction(self.G, self.N, xF, xU, self.device)
    
    def saveModels(self, path = "temp"):
        savedict = {'dimension': self.dimension,
            'layerConfig': self.layerConfig,
            'activation': self.activation,
            'G_state_dict': self.G.state_dict(),
            'N_state_dict': self.N.state_dict(),
           }
        Path(path).mkdir(parents=True, exist_ok=True)
        torch.save(savedict, path + "/model.pth")

    def checkSavedModels(self, loadPath):
        return Path(loadPath+"/model.pth").is_file()
        
    def loadModels(self, loadPath, device = device):
        model = torch.load(loadPath+'/model.pth')
        self.G = NN(numInputs = model['dimension']*2, numOutputs = model['dimension'], layerConfig = model['layerConfig'], activation = model['activation']).to(device)
        self.N = NN(numInputs = model['dimension'], numOutputs = model['dimension'], layerConfig = model['layerConfig'], activation = model['activation']).to(device)
        self.G.load_state_dict(model['G_state_dict'])
        self.N.load_state_dict(model['N_state_dict'])
        
         
        


