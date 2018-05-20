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

layer_name = 'fc8'

caffe.set_mode_cpu()
net = caffe.Net('bvlc/bvlc.prototxt',      # defines the structure of the model
                'bvlc/bvlc.caffemodel',  # contains the trained weights
                caffe.TEST)     # use test mode (e.g., don't perform dropout)

caffe_root = '/home/gin/caffe/'
mu = np.load(caffe_root + 'python/caffe/imagenet/ilsvrc_2012_mean.npy')
mu = mu.mean(1).mean(1)  # average over pixels to obtain the mean (BGR) pixel values
print 'mean-subtracted values:', zip('BGR', mu)

# create transformer for the input called 'data'
transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})

transformer.set_transpose('data', (2,0,1))  # move image channels to outermost dimension
transformer.set_mean('data', mu)            # subtract the dataset-mean value in each channel
transformer.set_raw_scale('data', 255)      # rescale from [0, 1] to [0, 255]
transformer.set_channel_swap('data', (2,1,0))  # swap channels from RGB to BGR

image = caffe.io.load_image('cornell_cropped.jpg')
input = transformer.preprocess('data', image)
from skimage import img_as_ubyte
net.blobs['data'].data[...] = input

### perform classification
print (net.layers)
output = net.forward(end=layer_name)
data = output[layer_name]
print (data)

from keras.models import model_from_json
print (K.image_data_format())
model = load_model('bvlc/new.h5', custom_objects={"LRN" : LRN})

model.summary()
inp = model.input

for layer in model.layers:
    if layer.name == layer_name:
        layer_output = K.function([inp]+ [K.learning_phase()], [layer.output])
        weights = layer.get_weights()

predictions = layer_output([input[np.newaxis], 0])
print (np.mean(np.abs(data - predictions)))

