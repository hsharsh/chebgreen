from .backend import torch, np, scipy, ABC, config, device

class Dataset(torch.utils.data.Dataset):
    def __init__(self, F, U):
        self.F = F
        self.U = U
    
    def __len__(self):
        return self.U.shape[0]
    
    def __getitem__(self, index):
        return self.F[index, :], self.U[index, :]
    
class DataProcessor(ABC):
    def __init__(self, filePath, seed = 42):
        self.data = None
        self.filePath = filePath
        self.seed = seed
        
    
    def generateDataset(self, trainRatio = 0.8, batch_size = 32):
        data = scipy.io.loadmat(self.filePath)
        np.random.seed(self.seed)

        self.xF = data['Y'].astype(dtype = config(np))
        self.xU = data['X'].astype(dtype = config(np))
        # self.xG = data['XG'].astype(dtype = config(np))
        # self.yG = data['YG'].astype(dtype = config(np))

        # Parameters
        params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 6}
        F = torch.from_numpy(data['F'].astype(dtype = config(np)))
        U = torch.from_numpy(data['U'].astype(dtype = config(np)))
        # Train-validation split
        iSplit = int(trainRatio*data['F'].shape[1])
        trainDataset = Dataset(F[:,:iSplit].T, U[:,:iSplit].T)
        self.trainDataset = torch.utils.data.DataLoader(trainDataset, **params)

        valDataset = Dataset(F[:,iSplit:].T, U[:,iSplit:].T)
        self.valDataset = torch.utils.data.DataLoader(valDataset, **params)

def generateEvaluationGrid(xU, xF):
    nF, nU, d = xF.shape[0], xU.shape[0], xU.shape[1]
    x, y = [],[]
    for i in range(d):
        x.append(torch.reshape(torch.tile(xU[:,i].reshape((1,nU)), [nF,1]), (nF*nU,1)))
        y.append(torch.reshape(torch.tile(xF[:,i].reshape((nF,1)), [1,nU]), (nF*nU,1)))
    return torch.stack(x+y, dim = 1)[...,0]