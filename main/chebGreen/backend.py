import numpy as np
import os,sys,glob, platform
from abc import ABC
from pathlib import Path
import matplotlib.pyplot as plt

# tf.float64 doesn't work for Apple tf2
class Config(ABC):
    def __init__(self, precision):
        self.precision = precision
    
    def __call__(self, package):
        config = {
            32: {np: np.float32},
            64: {np: np.float64},
        }[self.precision]
        return config[package]
 
config = Config(32)

# Set the appropriate matlab path according to system
operatingSystem = platform.uname().system
if operatingSystem == "Darwin":
    pathList = glob.glob("/Applications/MATLAB*.app/bin/matlab")
    if pathList:
        MATLABPath = pathList[0]
    else:
        raise('Matlab not found')
elif operatingSystem == "Linux":
    pathList = glob.glob("/usr/local/MATLAB/*/bin/matlab")
    if pathList:
        MATLABPath = pathList[0]
    else:
        raise('Matlab not found')
    