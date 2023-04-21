import scipy
import numpy as np
import logging
from abc import ABC

class DataProcessor(ABC):
    def __init__(self, filePath, seed = 42):
        self.data = None
        self.filePath = filePath
        self.seed = seed
        
    
    def trainTestSplit(self, testRatio = 0.8):
        data = scipy.io.loadmat(self.filePath)
        np.random.seed(self.seed)

        self.X = data['X']
        self.Y = data['Y']
