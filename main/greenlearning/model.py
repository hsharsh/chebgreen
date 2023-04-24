import tensorflow as tf
from .activations import get_activation
from .loss import LossGreensFunction
import numpy as np
from abc import ABC

class NN(tf.keras.Model):
    def __init__(self, numInputs = 2, numOutputs = 1, layerConfig = np.array([50, 50, 50, 50]), activation = "rational"):
        super().__init__()
        
        self.modelLayers = []
        # Input Layer
        self.modelLayers.append(tf.keras.layers.InputLayer(input_shape = (numInputs,)))
        
        for units in layerConfig:
            self.modelLayers.append(tf.keras.layers.Dense(units = units, activation = get_activation(activation)))
        
        self.modelLayers.append(tf.keras.layers.Dense(units = numOutputs, activation = None))
    
    def call(self, inputs, training=None, mask=None):
        x = self.modelLayers[0](inputs)
        for layer in self.modelLayers[1:]:
            x = layer(x)
        return x

class Model(ABC):
    def __init__(self, dimension = 1) -> None:
        self.epochs = int(1e3)

        self.G = NN(numInputs = dimension*2, numOutputs = dimension, layerConfig = np.array([50, 50, 50, 50]), activation = "rational")
        self.N = NN(numInputs = dimension, numOutputs = dimension, layerConfig = np.array([50, 50, 50, 50]), activation = "rational")

        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
        self.G.compile(optimizer = self.optimizer)
        self.N.compile(optimizer = self.optimizer)
        self.G.trainable_weights
    
    def train(self, data):
        self.init_loss(data.xF, data.xU)

        for epoch in range(self.epochs):
            for step, (fTrain, uTrain) in enumerate(data.train_dataset):

                with tf.GradientTape() as tape:
                    loss_value = self.lossfn(fTrain,uTrain)
                
                gradG, gradN = tape.gradient(loss_value, [self.G.trainable_weights, self.N.trainable_weights])
                self.optimizer.apply_gradients(zip(gradG, self.G.trainable_weights))
                self.optimizer.apply_gradients(zip(gradN, self.N.trainable_weights))

                print(f"Training loss at step {step} = {float(loss_value)}")


    
    def init_loss(self, xF, xU):
        self.lossfn = LossGreensFunction(self.G, self.N, xF, xU)
        


