import keras.backend as K
from keras.engine.topology import Layer
from keras.utils import conv_utils

class LRN(Layer):
    def __init__(self, alpha=1e-4, k=1, beta=0.75, n=5, data_format = None, **kwargs):
        if n % 2 == 0:
            raise NotImplementedError("LRN2D only works with odd n. n provided: " + str(n))
        super(LRN, self).__init__(**kwargs)
        self.alpha = alpha
        self.k = k
        self.beta = beta
        self.n = n
	self.data_format = conv_utils.normalize_data_format(data_format)

    def build(self, input_shape):
        super(LRN, self).build(input_shape)  # Be sure to call this somewhere!

    def call(self, X, mask=None):
	if self.data_format == 'channels_last':
        	b, r, c, ch = K.int_shape(X)
	else:
		b, ch, r, c = K.int_shape(X)
        half = self.n // 2
        square = K.square(X)
	if self.data_format != 'channels_last':
		extra_channels = K.spatial_2d_padding(square,
		                                      padding = ((half, half), (0, 0)), data_format='channels_last')
	else:
		extra_channels = K.spatial_2d_padding(K.permute_dimensions(square, (0, 3, 1, 2)),
							padding = ((half, half), (0, 0)), data_format='channels_last')
		
        scale = self.k
        for i in range(self.n):
            scale += (self.alpha / self.n) * extra_channels[:, i:(i + ch), :, :]
        scale = scale ** self.beta
	if self.data_format == 'channels_last':
	    scale = K.permute_dimensions(scale, (0, 2, 3, 1))
        return X / scale

    def compute_output_shape(self, input_shape):
        return input_shape

    def get_config(self):
        return {"name": self.name,
                "alpha": self.alpha,
                "k": self.k,
                "beta": self.beta,
                "n": self.n,
		"data_format": self.data_format}
