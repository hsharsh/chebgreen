from .backend import tf, ABC, config
from .utils import generateEvaluationGrid
from .quadrature_weights import get_weights

class LossGreensFunction(ABC):
    def __init__(self, G, N, xF, xU) -> None:
        super().__init__()
        self.G = G
        self.N = N
        self.xF, self.xU = xF, xU
        self.wF = tf.constant(get_weights('uniform', xF).T, dtype = config(tf))
        self.wU = tf.constant(get_weights('uniform', xU).T, dtype = config(tf))
    
    @tf.function
    def __call__(self, fTrain, uTrain):
        X = generateEvaluationGrid(self.xF, self.xU)
        nF, nU = self.xF.shape[0], self.xU.shape[0]
        G = tf.transpose(tf.reshape(self.G(X),(nF, nU)))
        u_hom = self.N(self.xU)

        uPred = tf.transpose(tf.matmul(G, tf.multiply(fTrain, self.wF), transpose_b = True) + u_hom)
        loss = tf.divide(tf.math.reduce_sum(tf.multiply(tf.square(uTrain - uPred),self.wU), axis = 1), \
                         tf.math.reduce_sum(tf.multiply(tf.square(uTrain),self.wU), axis = 1))
        return tf.math.reduce_mean(loss)