from keras.models import load_model, model_from_json
import numpy as np
import pylab as plt
from keras import backend as K
import caffe
import numpy as np
import sys

sys.path.append('caffe2keras/')
sys.path.append('caffe2keras/caffe2keras')
from lrn import LRN
from skimage.transform import resize
from skimage.io import imread
layer_name = "loss1/loss"

import numpy as np
import sys,os
import caffe


net = caffe.Net('ILGnet/deploy.prototxt',      # defines the structure of the model
                'ILGnet/ILGnet-AVA1.caffemodel',  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

caffe_root = '/home/gin/caffe/'
mu = np.load('ILGnet/mean/AVA2_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'

transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
transformer.set_transpose('data', (2,0,1))
transformer.set_mean('data', mu)
transformer.set_raw_scale('data', 255) 
transformer.set_channel_swap('data', (2,1,0))

image = caffe.io.load_image('cornell_cropped.jpg')#caffe_root + 'examples/images/fish-bike.jpg')
input = transformer.preprocess('data', image)
from skimage import img_as_ubyte
net.blobs['data'].data[...] = input


output = net.forward(end=layer_name)
data = output[layer_name]
print (data[0][1])


from keras.models import model_from_json

model = load_model('ILGnet/new.h5', custom_objects={"LRN" : LRN})

model.summary()
inp = model.input

for layer in model.layers:
    if layer.name == layer_name:
        layer_output = K.function([inp]+ [K.learning_phase()], [layer.output])
        weights = layer.get_weights()

predictions = layer_output([input[np.newaxis], 0])

print (np.mean(np.abs(data - predictions)))
print (np.mean(np.abs(data[0, 1] - predictions[0][0, 1])))
print (predictions)

