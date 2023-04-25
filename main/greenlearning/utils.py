from .backend import tf, np, scipy, ABC, config

buffer_size = 1024
class DataProcessor(ABC):
    def __init__(self, filePath, seed = 42):
        self.data = None
        self.filePath = filePath
        self.seed = seed
        
    
    def generateDataset(self, valRatio = 0.8, batch_size  = 32):
        data = scipy.io.loadmat(self.filePath)
        np.random.seed(self.seed)

        self.xF = data['Y'].astype(dtype = config(np))
        self.xU = data['X'].astype(dtype = config(np))
        self.xG = data['XG'].astype(dtype = config(np))
        self.yG = data['YG'].astype(dtype = config(np))
        self.u_hom = data['U_hom'].astype(dtype = config(np))

        # Train-validation split
        iSplit = int(valRatio*data['F'].shape[1])
        self.train_dataset = tf.data.Dataset.from_tensor_slices((data['F'][:,:iSplit].T.astype(dtype = config(np)), data['U'][:,:iSplit].T.astype(dtype = config(np))))
        self.train_dataset = self.train_dataset.shuffle(buffer_size = buffer_size).batch(batch_size)

        self.val_dataset = tf.data.Dataset.from_tensor_slices((data['F'][:,iSplit:].T.astype(dtype = config(np)), data['U'][:,iSplit:].T.astype(dtype = config(np))))
        self.val_dataset = self.val_dataset.batch(batch_size)