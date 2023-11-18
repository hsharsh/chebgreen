import numpy as np
import os,sys,glob,platform,configparser,ast,tempfile
from abc import ABC
from pathlib import Path
import matplotlib.pyplot as plt

# Initialize a parser for the settings.ini file
parser = configparser.ConfigParser(inline_comment_prefixes="#")
parser.read('settings.ini')

# Define a configuration class for handling precision settings
# tf.float64 doesn't work for Apple tf2
class Config(ABC):
    def __init__(self, precision):
        # Initialize the Config object with a precision setting
        self.precision = precision
    
    def __call__(self, package):
        # Define a dictionary mapping precision settings to numpy types
        config = {
            32: {np: np.float32},
            64: {np: np.float64},
        }[self.precision]
        # Return the numpy type corresponding to the precision setting
        return config[package]
 
# Create a Config object with the precision setting from the settings.ini file
config = Config(parser['GENERAL'].getint('precision'))

# Set the appropriate matlab path according to system
operatingSystem = platform.uname().system
if operatingSystem == "Darwin":
    # Search for the MATLAB application on MacOS
    pathList = glob.glob("/Applications/MATLAB*.app/bin/matlab")
    if pathList:
        # If found, set the MATLABPath to the first found path
        MATLABPath = pathList[0]
    else:
        # If not found, raise an error
        raise('Matlab not found')
elif operatingSystem == "Linux":
    # Search for the MATLAB application on Linux
    pathList = glob.glob("/usr/local/MATLAB/*/bin/matlab")
    if pathList:
        # If found, set the MATLABPath to the first found path
        MATLABPath = pathList[0]
    else:
        # If not found, raise an error
        raise('Matlab not found')
else:
    # If the operating system is not supported, raise an error
    raise('Operating system not supported. Please set MATLABPath manually.')
    