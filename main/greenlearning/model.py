import tensorflow as tf
from activation import Rational
import numpy as np

class GL(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.inputlayer = tf.keras.Input(shape = (4,))
        self.dense1 = tf.keras.layers.Dense(units = 50, activation = Rational())
        self.dense2 = tf.keras.layers.Dense(units = 50, activation = Rational())
        self.dense3 = tf.keras.layers.Dense(units = 50, activation = Rational())
        self.dense4 = tf.keras.layers.Dense(units = 50, activation = Rational())
        self.dense5 = tf.keras.layers.Dense(units = 50)
        self.outputlayer = tf.keras.layers.Dense(units = 1, activation = None)
    
    def call(self, inputs, training=None, mask=None):
        # x = self.inputlayer(inputs)
        x = self.dense1(inputs)
        x = self.dense2(x)
        x = self.dense3(x)
        x = self.dense4(x)
        x = self.dense5(x)
        out = self.outputlayer(x)
        return out