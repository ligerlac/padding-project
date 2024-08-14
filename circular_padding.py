import tensorflow as tf

class CircularPadding2D(tf.keras.layers.Layer):
    def __init__(self, padding=(1, 1), **kwargs):
        self.padding = padding
        super(CircularPadding2D, self).__init__(**kwargs)

    def build(self, input_shape):
        # No weights to initialize for this layer
        super(CircularPadding2D, self).build(input_shape)

    def call(self, inputs):
        pad_width, pad_height = self.padding

        # Circularly pad along the width (axis=2)
        x = tf.concat([inputs[:, :, -pad_width:], inputs, inputs[:, :, :pad_width]], axis=2)

        # Circularly pad along the height (axis=1)
        x = tf.concat([x[:, -pad_height:, :], x, x[:, :pad_height, :]], axis=1)

        return x

    def get_config(self):
        config = super(CircularPadding2D, self).get_config()
        config.update({"padding": self.padding})
        return config
