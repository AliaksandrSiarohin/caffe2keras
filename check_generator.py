from keras.models import load_model
import numpy as np
import pylab as plt
from keras import backend as K

import caffe


caffe_root = '/home/gin/caffe/'

mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values

model = load_model('generator/new.h5')
model.summary()

input = np.random.normal(size=(1, 4096))

layer_name = 'deconv0'

# for layer in model.layers:
#     if layer.name == layer_name + '_croping':
#         layer_output = K.function([inp]+ [K.learning_phase()], [layer.output])
#         weights = layer.get_weights()


# W_keras = weights[0]
# b_keras = weights[1]

keras_out = model.predict(input)

net = caffe.Net('generator/generator.prototxt',      # defines the structure of the model
                'generator/generator.caffemodel',  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)


net.blobs['feat'].data[...] = input


output = net.forward(end=layer_name)
caffe_out = output[layer_name]

print (np.sum(np.abs(keras_out - caffe_out)))
