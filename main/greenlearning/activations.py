from .backend import tf, config

class Rational(tf.keras.layers.Layer):
    """
    Rational Activation function.
    f(x) = P(x) / Q(x)
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self, trainable=True, name=None, dtype=config(tf), dynamic=False, **kwargs):
        super().__init__(trainable, name, dtype, dynamic, **kwargs)
        self.coeffs = self.add_weight("Numerator coeffs", shape = [4,2], dtype = config(tf), initializer=self.Coefficient_Initializer)
        self.mask = tf.transpose(tf.constant([[1., 1., 1., 1.], [0., 1., 1., 1.]], dtype = config(tf)))
        
    def call(self, inputs, *args, **kwargs):
        exponents = tf.constant([3., 2., 1. , 0.], dtype = config(tf))
        X = tf.pow(tf.expand_dims(inputs, axis = -1), exponents)
        PQ = X @ (self.coeffs*self.mask)
        return tf.divide(PQ[...,0],PQ[...,1])
    class Coefficient_Initializer(tf.keras.initializers.Initializer):
        """
        Initializer for the coefficients of the rational function
        """
        def __init__(self) -> None:
            super().__init__()
        
        def __call__(self, shape, dtype = config(tf), **kwargs):
            return tf.transpose(tf.constant([[1.1915, 1.5957, 0.5, 0.0218], [0., 2.383, 0.0, 1.0]], dtype = dtype))

def get_activation(identifier):
    """Return the activation function."""
    if isinstance(identifier, str):
        return {
                'elu': tf.nn.elu,
                'relu': tf.nn.relu,
                'selu': tf.nn.selu,
                'sigmoid': tf.nn.sigmoid,
                'sin': tf.sin,
                'tanh': tf.nn.tanh,
                'rational': Rational(),
                }[identifier]