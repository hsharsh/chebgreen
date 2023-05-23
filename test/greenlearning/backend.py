import numpy as np
import torch
import scipy
from abc import ABC
from pathlib import Path

# tf.float64 doesn't work for Apple tf2
class Config(ABC):
    def __init__(self, precision):
        self.precision = precision
    
    def __call__(self, package):
        config = {
            32: {np: np.float32, torch: torch.float},
            64: {np: np.float64, torch: torch.double},
        }[self.precision]
        return config[package]
 
config = Config(32)

# Set the appropriate compute acceleration, default to CPU if nothing is available
if torch.backends.mps.is_available():
    device = torch.device('mps')
elif torch.cuda.is_available():
    device = torch.device('cuda:0')
else:
    device = torch.device('cpu')
