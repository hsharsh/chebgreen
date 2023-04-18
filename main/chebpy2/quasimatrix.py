import chebpy
import numpy as np
from abc import ABC, abstractmethod, abstractclassmethod

class Quasimatrix(ABC):
    """Create a Quasimatrix in order to implement most functionality for the chebfun2 constructor"""
    def __init__(self, data, transposed = False):
        # Currently only initialized with a numpy array of chebfuns
        self.data = np.array(data).reshape(-1)
        if self.data is not None:
            self.domain = self.data[0].domain
            self.prefs = self.data[0].prefs
        self.transposed = transposed
    
    # Disable matrix multiplication from numpy with broadcasting while right multiplying.
    __array_ufunc__ = None

    def __repr__(self):
        if not self.transposed:
            header = f"Quasimatrix of shape ({self.shape[0]} x {self.shape[1]}) with columns:\n"
        else:
            header = f"Quasimatrix of shape ({self.shape[0]} x {self.shape[1]}) with rows:\n"
        return header + self.data.__repr__()
            
    def __getitem__(self, key):
        x, y = key
        
        if not self.transposed:
            assert isinstance(y,slice) or isinstance(y,int), 'Second index needs to be a slice or an integer'

            if isinstance(x,int) or isinstance(x,float):
                return np.array([col(x) for col in self.data[y]])
            elif isinstance(x,np.ndarray) and (x.dtype == np.int64 or x.dtype == np.float64):
                return np.array([col(x) for col in self.data[y]]).T
            elif (x == slice(None)):
                return Quasimatrix(data = np.array(self.data[y]).reshape(-1), transposed = self.transposed)
            else:
                raise RuntimeError('The first index needs to be a float or a numpy array of floats')
        else:
            assert isinstance(x,slice), 'First index needs to be a slice'
            if isinstance(y,int) or isinstance(y,float):
                return np.array([row(y) for row in self.data[x]])
            elif isinstance(y,np.ndarray) and (y.dtype == np.int64 or y.dtype == np.float64):
                return np.array([row(y) for row in self.data[x]])
            elif y == slice(None):
                return Quasimatrix(data = np.array(self.data[x]).reshape(-1), transposed = self.transposed)
            else:
                raise RuntimeError('The second index needs to be a float or a numpy array of floats')
    
    def __add__(self, qmat):
        # Addition of quasimatrices
        assert self.shape == qmat.shape, f"Cannot add a ({self.shape[0]} x {self.shape[1]}) matrix to a ({qmat.shape[0]} x {qmat.shape[1]}) matrix."
        return Quasimatrix(data = self.data + qmat.data, transposed = self.transposed)

    # def __mul__(self, qmat):
    #     # Multiplication of quasimatrices
    #     assert self.shape[1] == qmat.shape[0], f"Cannot mulitply a ({self.shape[0]} x {self.shape[1]}) matrix with a ({qmat.shape[0]} x {qmat.shape[1]}) matrix."
        
    #     if self.shape[1] == np.inf:
    #         mat = np.zeros((self.shape[0],qmat.shape[1]))
    #         for i in range(self.shape[0]):
    #             for j in range(qmat.shape[1]):
    #                 mat[i,j] = chebpy.core.algorithms.innerproduct(self.data[i],qmat.data[j])
    #     else:
    #         raise NotImplementedError

    def __mul__(self, G):
        """
        Multiplication of quasimatrices

        Here, I am assuming that the quasimatrices only comprises of chebfuns which
        are of the same rank within a quasimatrix and that each chebfun only has one
        single interval.
        Faster but uses more space.
        """
        F = self 

        if isinstance(G,int) or isinstance(G,float):
            return Quasimatrix(F.data * G, transposed = F.transposed)

        # Check if the inputs can be multiplied
        assert F.shape[1] == G.shape[0], f"Cannot mulitply a ({F.shape[0]} x {F.shape[1]}) matrix with a ({G.shape[0]} x {G.shape[1]}) matrix."
        
        if isinstance(G,np.ndarray) and (G.dtype == np.int64 or G.dtype == np.float64):
            if not F.transposed:
                return Quasimatrix(data = chebpy.chebfun(F.coeffrepr() @ G, domain = F.domain, prefs = F.prefs, initcoeffs = True), transposed = True) # Returns (inf x G.shape[1]) matrix
            else:
                RuntimeError('Invalid multiplication')  # This should never be reached because of shape check
        
        if F.shape[1] == np.inf: 
            assert F.domain == G.domain, "Cannot multiply quasimatrices on different domains"
            out = 0
            m, n = F.shape[0], G.shape[1]
            
            N = len(F.data[0].coeffs) + len(G.data[0].coeffs)
            Fvalues = np.zeros((m,N))
            Gvalues = np.zeros((N,n))
            
            prolongtemp = np.zeros(N)
            for i in range(m):
                f = F.data[i].funs[0]
                prolongtemp[:len(f.coeffs)] = f.coeffs
                f = f.onefun._coeffs2vals(prolongtemp)
                Fvalues[i,:len(f)] = f
            
            for j in range(n):
                g = G.data[j].funs[0]
                prolongtemp[:len(g.coeffs)] = g.coeffs
                g = g.onefun._coeffs2vals(prolongtemp)
                Gvalues[:len(g),j] = g
                
            w = chebpy.core.algorithms.quadwts2(N).reshape((1,-1))
            rescalingFactor = 0.5 * float(np.diff(F.domain))
            
            return (w * Fvalues) @ Gvalues * rescalingFactor
        else:
            raise NotImplementedError

    def __rmul__(self, F):
        """
        Multiplication of quasimatrices

        Here, I am assuming that the quasimatrices only comprises of chebfuns which
        are of the same rank within a quasimatrix and that each chebfun only has one
        single interval.
        Faster but uses more space.
        """
        G = self 

        if isinstance(F,int) or isinstance(F,float):
            return Quasimatrix(F * G.data, transposed = G.transposed)

        # Check if the inputs can be multiplied
        assert F.shape[1] == G.shape[0], f"Cannot mulitply a ({F.shape[0]} x {F.shape[1]}) matrix with a ({G.shape[0]} x {G.shape[1]}) matrix."

        if isinstance(F,np.ndarray) and (F.dtype == np.int64 or F.dtype == np.float64):
            if G.transposed:
                return Quasimatrix(data = chebpy.chebfun(F @ G.coeffrepr(), domain = G.domain, prefs = G.prefs, initcoeffs = True), transposed = False) # Returns (G.shape[0] x inf) matrix
            else:
                RuntimeError('Invalid multiplication')  # This should never be reached because of shape check
        
        if F.shape[1] == np.inf: 
            assert F.domain == G.domain, "Cannot multiply quasimatrices on different domains"
            out = 0
            m, n = F.shape[0], G.shape[1]
            
            N = len(F.data[0].coeffs) + len(G.data[0].coeffs)
            Fvalues = np.zeros((m,N))
            Gvalues = np.zeros((N,n))
            
            prolongtemp = np.zeros(N)
            for i in range(m):
                f = F.data[i].funs[0]
                prolongtemp[:len(f.coeffs)] = f.coeffs
                f = f.onefun._coeffs2vals(prolongtemp)
                Fvalues[i,:len(f)] = f
            
            for j in range(n):
                g = G.data[j].funs[0]
                prolongtemp[:len(g.coeffs)] = g.coeffs
                g = g.onefun._coeffs2vals(prolongtemp)
                Gvalues[:len(g),j] = g
                
            w = chebpy.core.algorithms.quadwts2(N).reshape((1,-1))
            rescalingFactor = 0.5 * float(np.diff(F.domain))
            
            return (w * Fvalues) @ Gvalues * rescalingFactor
        else:
            raise NotImplementedError        
    # def __rmul__(self,G):
    #     F = self
    #     if (isinstance(G,int) or isinstance(G,float)) or (isinstance(G,np.ndarray) and (G.dtype == np.int64 or G.dtype == np.float64)):
    #         return Quasimatrix(G * F.data, transposed = F.transposed)         
    ### Properties
    @property
    def shape(self):
        if self.transposed:
            return len(self.data), np.inf
        else:
            return np.inf, len(self.data)
        
    @property
    def T(self):
        # Tranpose of a quasimatrix
        return Quasimatrix(data = self.data, transposed = not self.transposed)       

    def coeffrepr(self):
        if len(self.data) == 0:
            raise RuntimeError('Cannot construct a coefficient representation for an empty quasimatrix')
        if self.transposed:
            return np.array([row.coeffs for row in self.data])
        else:
            return np.array([col.coeffs for col in self.data]).T
    ###  Utilities
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