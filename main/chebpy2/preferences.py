import numpy as np
from abc import ABC
import chebpy

class Chebpy2Preferences(ABC):
    """Default preferences for chebpy."""
    def __init__(self) -> None:
        super().__init__()
        self.minSample = np.array([17,17])
        self.maxRank = np.array([513,513])
        self.domain = np.array([-1.0, 1.0, -1.0, 1.0])
        
        self.prefx = chebpy.core.settings.ChebPreferences()
        self.prefy = chebpy.core.settings.ChebPreferences()
        self.prefx.eps = 1e-8
        self.prefy.eps = 1e-8

cheb2prefs = Chebpy2Preferences()