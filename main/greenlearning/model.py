from .backend import tf, np, ABC, config
from .utils import generateEvaluationGrid
from .activations import get_activation
from .loss import LossGreensFunction



def NN(numInputs = 2, numOutputs = 1, layerConfig = np.array([50, 50, 50, 50]), activation = "rational", dtype = config(tf)):
    layers = []
    layers.append(tf.keras.layers.InputLayer(input_shape = (numInputs,), dtype = config(tf)))
        
    for units in layerConfig:
        layers.append(tf.keras.layers.Dense(units = units, dtype = config(tf), activation = get_activation(activation)))
    
    layers.append(tf.keras.layers.Dense(units = numOutputs, dtype = config(tf), activation = None))
    return tf.keras.Sequential(layers)
class Model(ABC):
    def __init__(self, dimension = 1, layerConfig = [50, 50, 50, 50], activation = "rational") -> None:
        self.dimension = dimension
        self.G = NN(numInputs = dimension*2, numOutputs = dimension, layerConfig = np.array(layerConfig), activation = activation)
        self.N = NN(numInputs = dimension, numOutputs = dimension, layerConfig = np.array(layerConfig), activation = activation)
        
        self.epochs = int(1e3)
        self.optimizer = tf.keras.optimizers.Adam(learning_rate = 0.001, beta_1 = 0.9, beta_2 = 0.999, epsilon = 1e-8)
    
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

            print(f"Training loss at epoch {epoch} = {config(np)(loss_value):.2E}")
            
    
    # def __call__(self, xF, xU, weightsF):
    #     return self.G(x,x)

    def init_loss(self, xF, xU):
        self.lossfn = LossGreensFunction(self.G, self.N, xF, xU)
        


