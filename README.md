## Caffe to Keras converter

**Note:** This converter has been adapted from code in [Marc Bolaños fork of
Caffe]().

This is intended to serve as a conversion module for Caffe models to Keras
models. It only works
with [Ye Olde Caffe Classic™](https://github.com/BVLC/caffe) (which isn't really
a thing, but which probably should be a thing to prevent confusion with
the [Caffe 2](https://caffe2.ai/)).

Please be aware that this module is not regularly maintained. Thus, some layers
or parameter definitions introduced in newer versions of either Keras or Caffe
might not be compatible with the converter. Pull requests welcome!

### Conversion

In order to convert a model you just need the `.caffemodel` weights and the
`.prototxt` deploy file. You will need to include the input image dimensions as
a header to the `.prototxt` network structure, preferably as an `Input` layer:

```
layer {
  name: "image"
  type: "Input"
  top: "image"
  input_param {shape {dim: 1, dim: 3, dim: 128, dim: 128}}
}
```

Given the differences between Caffe and Keras when applying the max pooling
operation, in some occasions the max pooling layers must include a `pad: 1`
value even if they did not include them in their original `.prototxt`.

The module `caffe2keras` can be used as a command line interface for converting
any model the following way, tested only with keras=2.0.8, need theano backend for conversion.
After conversion any backend can be used:

```
KERAS_BACKEND=th python -m caffe2keras models/train_val_for_keras.prototxt models/bvlc_googlenet.caffemodel keras-output-model.h5
```

To use the produced model from Keras, simply load the output file (i.e.
`keras-output-model.h5`) using `keras.models.load_model`.

By default all models saved in channels_first format, if you need channels last format use:
```
KERAS_BACKEND=th python -m caffe2keras --backend tf models/train_val_for_keras.prototxt models/bvlc_googlenet.caffemodel keras-output-model.h5
```
This will just replace channels_first to channesl_last in .json file. Use this file to load model, and then load weights to it.
### Succesfully converted models

[Realtime Multi-Person Pose Estimation](https://github.com/ZheC/Realtime_Multi-Person_Pose_Estimation)

[BVLC reference caffenet](https://github.com/BVLC/caffe/tree/master/models/bvlc_reference_caffenet)
(*Note there is not LRN Layer in keras, copy lrn layer to your project from caffe2keras folder. Also for some reason vesrion for convolution and fc layers not detected correctly in convert.py, force the version to V1 (line 642 and 603).)

[Generator network from PPGN](https://github.com/Evolving-AI-Lab/ppgn)
(Not exactly correct)

[ILGNET](https://github.com/BestiVictory/ILGnet)
(*Note this model converts some pooling in a wrong way, check "crutch" in convert.py.)

[MEMNET](http://memorability.csail.mit.edu/download.html)


Use file similar to ```check_bvlc.py``` for checking the correctness of conversion.
Usually it never convert model from the first time and different hacks is needed to make it work.

### Acknowledgments

This code is yet another iteration of a tool which many people have contributed
to. Previous authors:

- Marc Bolaños ([email](mailto:marc.bolanos@ub.edu), [Github](https://github.com/MarcBS))
- Pranav Shyam ([Github](https://github.com/pranv))
- Antonella Cascitelli ([Github](https://github.com/lenlen))
