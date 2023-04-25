from .backend import tf, ABC, config
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
        X = self.generateEvaluationGrid(self.xF, self.xU)
        nF, nU = fTrain.shape[1], uTrain.shape[1]
        G = tf.transpose(tf.reshape(self.G(X),(nF, -1)))
        u_hom = self.N(self.xU)

        uPred = tf.transpose(tf.matmul(G, tf.multiply(fTrain, tf.constant(self.wF)), transpose_b = True) + u_hom)
        loss = tf.math.reduce_sum(tf.multiply(uTrain - uPred,self.wU), axis = 1)/tf.math.reduce_sum(tf.multiply(uTrain,self.wU), axis = 1)
        return tf.math.reduce_mean(loss)
    
    def generateEvaluationGrid(self, xF, xU):
        nF, nU, d = xF.shape[0], xU.shape[0], xU.shape[1]
        x, y = [],[]
        for i in range(d):
            x.append(tf.reshape(tf.tile(xU[:,i].reshape((1,nU)), [nF,1]), (nF*nU,1)))
            y.append(tf.reshape(tf.tile(xF[:,i].reshape((nF,1)), [1,nU]), (nF*nU,1)))
        return tf.concat(x+y, axis = 1)