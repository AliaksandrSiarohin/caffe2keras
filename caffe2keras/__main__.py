#!/usr/bin/env python
"""Run this with ``python -m caffe2keras <args>``"""

import argparse

parser = argparse.ArgumentParser(
    description='Converts a Caffe model to a Keras model')
parser.add_argument(
    '--debug', action='store_true', default=False, help='enable debug mode')
parser.add_argument('prototxt', help='network definition path')
parser.add_argument('caffemodel', help='network weights path')
parser.add_argument('destination', help='path for output model')
parser.add_argument('--backend', default='th', help='Store in model defenition in this backend')

def main():
    args = parser.parse_args()

    # lazy import so that we parse args before initialising TF/Theano
    from caffe2keras import convert

    print("Converting model...")
    model = convert.caffe_to_keras(args.prototxt, args.caffemodel, args.debug)
    print("Finished converting model")

    # Save converted model structure
    print("Storing model...")
    model.save(args.destination)
    json_s = model.to_json()
    if args.backend == 'tf':
	json_s = json_s.replace('channels_first', 'channels_last')
        json_s = json_s.replace('theano', 'tensorflow')
    import json
    parsed = json.loads(json_s)
    json_s = json.dumps(parsed, indent=4, sort_keys=True)
    with open(args.destination.replace('h5', 'json'), 'w') as f:
	print >>f, json_s
    print("Finished storing the converted model to " + args.destination)


if __name__ == '__main__':
    main()
