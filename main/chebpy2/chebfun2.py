import chebpy as cp
import numpy as np
from abc import ABC, abstractmethod, abstractclassmethod

class Chebfun2(ABC):
    def __init__(self,g = None, domain = np.array([-1, 1, -1, 1])):

        # Default domain is [-1,1] x [-1, 1]
        self.domain = domain
        self.cols, self.rows, self.pivotValues, self.rank, self.vscale, self.cornervalues = self.constructor(g)
    
    def __repr__(self):
        header = f"chebfun2 object\n"
        toprow = "     domain       rank     corner values\n"
        rowdta = (f"[{self.domain[0]},{self.domain[1]}] x [{self.domain[2]},{self.domain[3]}]     {self.rank}       "
            f"[{self.cornervalues[0]} {self.cornervalues[1]} {self.cornervalues[2]} {self.cornervalues[3]}]\n")
        btmrow = f"vertical scale = {self.vscale}"
        return header + toprow + rowdta + btmrow
    

    def constructor(self, g = None):
        cols = np.array([])    # This should be a quasimatrix
        rows = np.array([])    # This should be a quasimatrix
        pivotValues = np.array([])
        rank  = len(pivotValues)
        vscale = 0
        cornervalues = np.array([0,0,0,0])

        return cols, rows, pivotValues, rank, vscale, cornervalues