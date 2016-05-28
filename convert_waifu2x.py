import sys
import json
import math
import os

import tensorflow as tf
import numpy as np

top_dir = os.path.normpath(os.path.join(__file__, '..'))
waifu2x_dir = os.path.join(top_dir, 'waifu2x')
model_file = os.path.join(waifu2x_dir, 'models/anime_style_art_rgb/scale2.0x_model.json')
#model_file = os.path.join(waifu2x_dir, 'models/photo/scale2.0x_model.json')

def conv2d(x, params, name=None):
    in_channels = params['nInputPlane']
    out_channels = params['nOutputPlane']
    filter_width = params['kW']
    filter_height = params['kH']
    W_data = np.float32(params['weight']).transpose(2, 3, 1, 0)
    b_data = np.float32(params['bias'])

    W = tf.constant(W_data, name="W_{}".format(name))
    b = tf.constant(b_data, name="b_{}".format(name))

    #padding = 'VALID'
    padding = 'SAME'
    conv = tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding=padding, name=name)
    return tf.nn.bias_add(conv, b, name="out_{}".format(name))

def leaky_relu(x, alpha, name=None):
    return tf.maximum(x, tf.scalar_mul(alpha, x), name=name)

def convert_model(out_dir):
    with open(model_file) as fp:
        model_params = json.load(fp)

    g = tf.Graph()
    with g.as_default():
        h = x = tf.placeholder("float", shape=[1, 128, 128, 3], name="x")
        steps = len(model_params)
        for i, layer_params in enumerate(model_params):
            name = "conv{}".format(i + 1)
            print(name)
            h = conv2d(h, layer_params, name=name)
            if i < steps - 1:
                h = leaky_relu(h, 0.2, name="leaky_relu_{}".format(name))

        session = tf.Session()
        init = tf.initialize_all_variables()
        session.run(init)

        graph_def = g.as_graph_def()
        tf.train.write_graph(graph_def, out_dir, 'tf_waifu2x.pb', as_text=False)

convert_model(os.getcwd())
