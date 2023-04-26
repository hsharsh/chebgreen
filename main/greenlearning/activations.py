from .backend import tf, np, config

class Rational(tf.keras.layers.Layer):
    """
    Rational Activation function.
    f(x) = P(x) / Q(x)
    where the coefficients of P and Q are initialized to the best rational 
    approximation of degree (3,2) to the ReLU function
    # Reference
        - [Rational neural networks](https://arxiv.org/abs/2004.01902)
    """
    def __init__(self,
                trainable: bool = True,
                **kwargs):
        super().__init__(**kwargs)
        self.trainable = trainable

    def build(self, input_shape: tf.TensorShape, dtype = config(tf)):
        super().build(input_shape)  
        self.Pcoeffs = tf.unstack(tf.Variable(
            [1.1915, 1.5957, 0.5, 0.0218],
            dtype=dtype,
            trainable=self.trainable,
            name="Pcoeffs"))
        self.Qcoeffs = tf.unstack(tf.Variable(
            [2.383, 0.0, 1.0],
            dtype=dtype,
            trainable=self.trainable,
            name="Pcoeffs"))


    def call(self,
            inputs: tf.Tensor
            ) -> tf.Tensor:
        return tf.math.divide(tf.math.polyval(self.Pcoeffs,inputs), tf.math.polyval(self.Qcoeffs,inputs))

    def get_config(self):
        config = {
            "P Coefficients": self.get_weights()[0],
            "Q coefficients": self.get_weights()[1],
            "trainable": self.trainable
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))

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