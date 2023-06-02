from .backend import torch, np, ABC, config, device
from .utils import generateEvaluationGrid, approximateDistanceFunction
from .quadrature_weights import get_weights

class LossGreensFunction(ABC):
    def __init__(self, G, N, xF, xU, domain, device) -> None:
        super().__init__()
        self.G = G
        self.N = N
        self.xF, self.xU = torch.from_numpy(xF).to(device), torch.from_numpy(xU).to(device)
        self.wF = torch.from_numpy(get_weights('uniform', xF).astype(dtype = config(np)).T).to(device)
        self.wU = torch.from_numpy(get_weights('uniform', xU).astype(dtype = config(np)).T).to(device)
        self.X = generateEvaluationGrid(self.xU, self.xF).to(device)
        self.ADF = approximateDistanceFunction(self.X[:,0], self.X[:,1],
                                               domain.astype(config(np)),
                                               device
                                               ).to(device).reshape((-1,1))

    def __call__(self, fTrain, uTrain):
        nF, nU = self.xF.shape[0], self.xU.shape[0]
        G = torch.reshape(self.ADF*self.G(self.X),(nF, nU)).T
        u_hom = self.N(self.xU)

        uPred = (torch.matmul(G, torch.mul(fTrain, self.wF).T) + u_hom).T
        
        loss = torch.sum(torch.square(uTrain - uPred)*self.wU, dim = 1)/ \
                torch.sum(torch.square(uTrain)*self.wU, dim = 1)
        return torch.mean(loss)