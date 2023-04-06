import chebpy
import numpy as np
from abc import ABC, abstractmethod, abstractclassmethod

class Quasimatrix(ABC):
    """Create a Quasimatrix in order to implement most functionality for the chebfun2 constructor"""
    def __init__(self, data, transposed = False):
        # Currently only initialized with a numpy array of chebfuns
        self.data = data
        self.transposed = transposed

    def __getitem__(self, key):
        x, y = key
        
        if not self.transposed:
            assert isinstance(y,slice) or isinstance(y,int), 'Second index needs to be a slice or an integer'

            if isinstance(x,int) or isinstance(x,float):
                return np.array([col(x) for col in self.data[y]])
            elif isinstance(x,np.ndarray) and (x.dtype == np.int64 or x.dtype == np.float64):
                return np.array([col(x) for col in self.data[y]])
            elif (x == slice(None)):
                return Quasimatrix(data = self.data[y], transposed = self.transposed)
            else:
                raise RuntimeError('The first index needs to be a float or a numpy array of floats')
        else:
            assert isinstance(x,slice), 'First index needs to be a slice'
            if isinstance(y,int) or isinstance(y,float):
                return np.array([row(y) for row in self.data[x]])
            elif isinstance(y,np.ndarray) and (y.dtype == np.int64 or y.dtype == np.float64):
                return np.array([row(y) for row in self.data[x]]).T
            elif y == slice(None):
                return Quasimatrix(data = self.data[x], transposed = self.transposed)
            else:
                raise RuntimeError('The second index needs to be a float or a numpy array of floats')
            
    def plot(self, ax=None, **kwds):
        if isinstance(self.data,chebpy.core.chebfun.Chebfun):
            self.data.plot(ax = None, **kwds)
        else:
            for i in range(len(self.data)):
                self.data[i].plot(ax = None, **kwds)
    
    def plotcoeffs(self, ax=None, **kwds):
        if isinstance(self.data,chebpy.core.chebfun.Chebfun):
            self.data.plotcoeffs(ax = None, **kwds)
        else:
            for i in range(len(self.data)):
                self.data[i].plotcoeffs(ax = None, **kwds)