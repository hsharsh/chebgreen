import tensorflow as tf
from .quadrature_weights import get_weights
from abc import ABC

class LossGreensFunction(ABC):
    def __init__(self, G, N, xF, xU) -> None:
        super().__init__()
        self.G = G
        self.N = N
        self.xF = xF
        self.xU = xU
    
    def __call__(self, fTrain, uTrain):
        weightsF = get_weights('uniform', self.xF)
        weightsU = get_weights('uniform', self.xU)


        return tf.math.reduce_mean(uTrain)
