import numpy as np
import tensorflow as tf
import scipy
from abc import ABC

# tf.float64 doesn't work for Apple tf2
class Config(ABC):
    def __init__(self, precision):
        self.precision = precision
    
    def __call__(self, package):
        config = {
            32: {np: np.float32, tf: tf.float32},
            64: {np: np.float64, tf: tf.float64},
        }[self.precision]
        return config[package]
 
config = Config(32)

# Need to explicitly set the config for float to float64 since tf.keras.Model.__call__ casts to tf.backend.floatx
if config(tf) == tf.float64:
    tf.keras.backend.set_floatx('float64')
