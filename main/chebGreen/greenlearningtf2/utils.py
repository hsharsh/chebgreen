from .backend import tf, np, scipy, ABC, config

buffer_size = 1024
class DataProcessor(ABC):
    def __init__(self, filePath, seed = 42):
        self.data = None
        self.filePath = filePath
        self.seed = seed
        
    
    def generateDataset(self, trainRatio = 0.8, batch_size  = 1024):
        data = scipy.io.loadmat(self.filePath)
        np.random.seed(self.seed)

        self.xF = data['Y'].astype(dtype = config(np))
        self.xU = data['X'].astype(dtype = config(np))
        self.xG = data['XG'].astype(dtype = config(np))
        self.yG = data['YG'].astype(dtype = config(np))
        self.u_hom = data['U_hom'].astype(dtype = config(np))

        # Train-validation split
        iSplit = int(trainRatio*data['F'].shape[1])
        self.trainDataset = tf.data.Dataset.from_tensor_slices((data['F'][:,:iSplit].T.astype(dtype = config(np)), data['U'][:,:iSplit].T.astype(dtype = config(np))))
        self.trainDataset = self.trainDataset.shuffle(buffer_size = buffer_size).batch(batch_size)

        self.valDataset = tf.data.Dataset.from_tensor_slices((data['F'][:,iSplit:].T.astype(dtype = config(np)), data['U'][:,iSplit:].T.astype(dtype = config(np))))
        self.valDataset = self.valDataset.batch(batch_size)

def generateEvaluationGrid(xU, xF):
    nF, nU, d = xF.shape[0], xU.shape[0], xU.shape[1]
    x, y = [],[]
    for i in range(d):
        x.append(tf.reshape(tf.tile(xU[:,i].reshape((1,nU)), [nF,1]), (nF*nU,1)))
        y.append(tf.reshape(tf.tile(xF[:,i].reshape((nF,1)), [1,nU]), (nF*nU,1)))
    return tf.concat(x+y, axis = 1)

def approximateDistanceFunction(x, y, domain):
    def distance(x1, y1, x2, y2):
        return tf.math.sqrt(tf.square(x2-x1) + tf.square(y2-y1))

    def lineSegment(x, y, x1, y1, x2, y2):
        L = distance(x1, y1, x2, y2)
        xc, yc = (x1+x2)/2, (y1+y2)/2
        f = (1/L) * ((x-x1)*(y2-y1) - (y-y1) * (x2-x1))
        t = (1/L) * ((L/2)**2 - tf.square(distance(x,y,xc,yc)))
        phi = tf.math.sqrt(tf.square(t) + tf.math.pow(f,4))
        return tf.math.sqrt(tf.square(f) + 0.25 * tf.square(phi-t))
    
    R = tf.zeros((x.shape[0],1))
    segments = tf.constant(np.array([[domain[0], domain[2], domain[1], domain[2]],
                         [domain[1], domain[2], domain[1], domain[3]],
                         [domain[1], domain[3], domain[0], domain[3]],
                         [domain[0], domain[3], domain[0], domain[2]]
                        ]), dtype = config(tf))
    phi = []
    for i in range(4):
        phi.append(lineSegment(x,y,segments[i,0], segments[i,1], segments[i,2], segments[i,3]))
    Phi = phi[0]*phi[1]*phi[2]*phi[3]/(phi[0]+phi[1]+phi[2]+phi[3])
    
    return Phi