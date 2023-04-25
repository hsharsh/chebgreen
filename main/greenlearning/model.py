from .backend import tf, np, ABC, config
from .activations import get_activation
from .loss import LossGreensFunction


# Need to explicitly set the config for float to float64 since tf.keras.Model.__call__ casts to tf.backend.floatx


class NN(tf.keras.Model):
    def __init__(self, numInputs = 2, numOutputs = 1, layerConfig = np.array([50, 50, 50, 50]), activation = "rational", dtype = config(tf)):
        super().__init__()
        
        self.modelLayers = []
        # Input Layer
        self.modelLayers.append(tf.keras.layers.InputLayer(input_shape = (numInputs,), dtype = config(tf)))
        
        for units in layerConfig:
            self.modelLayers.append(tf.keras.layers.Dense(units = units, dtype = config(tf), activation = get_activation(activation)))
        
        self.modelLayers.append(tf.keras.layers.Dense(units = numOutputs, dtype = config(tf), activation = None))
    
    def call(self, inputs, training=None, mask=None):
        # print(inputs.dtype)
        x = self.modelLayers[0](inputs)
        for layer in self.modelLayers[1:]:
            x = layer(x)
        return x

class Model(ABC):
    def __init__(self, dimension = 1) -> None:
        self.dimension = dimension
        self.G = NN(numInputs = dimension*2, numOutputs = dimension, layerConfig = np.array([50, 50, 50, 50]), activation = "rational")
        self.N = NN(numInputs = dimension, numOutputs = dimension, layerConfig = np.array([50, 50, 50, 50]), activation = "rational")

        self.epochs = int(1e3)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
        self.G.compile(optimizer = self.optimizer)
        self.N.compile(optimizer = self.optimizer)
    
    def train(self, data):
        
        assert data.xF.shape[1] == self.dimension and data.xU.shape[1] == self.dimension, f"Dimension of \
            evaluation points for forcing, {data.xF.shape[1]}, and response, {data.xU.shape[1]}, should \
            match with the model dimension, {self.dimension}"
        
        self.init_loss(data.xF, data.xU)

        for epoch in range(self.epochs):
            for step, (fTrain, uTrain) in enumerate(data.train_dataset):
                with tf.GradientTape() as tape:
                    loss_value = self.lossfn(fTrain,uTrain)
                    
                gradG, gradN = tape.gradient(loss_value, [self.G.trainable_weights, self.N.trainable_weights])
                self.optimizer.apply_gradients(zip(gradG, self.G.trainable_weights))
                self.optimizer.apply_gradients(zip(gradN, self.N.trainable_weights))

                print(f"Training loss at epoch {epoch}, step {step} = {float(loss_value)}")
            
    
    def __call__(self, f, x, weightsF):
        self.G(x,x)

    def init_loss(self, xF, xU):
        self.lossfn = LossGreensFunction(self.G, self.N, xF, xU)
        


