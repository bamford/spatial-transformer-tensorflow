# Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
from scipy import ndimage
import tensorflow.compat.v1 as tf
from spatial_transformer import ProjectiveTransformer
import numpy as np
import scipy.misc

# Input image retrieved from:
# https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
im = ndimage.imread('data/cat.jpg')
im = im / 255.
im = im.astype('float32')

# input batch
batch_size = 4
batch = np.expand_dims(im, axis=0)
batch = np.tile(batch, [batch_size, 1, 1, 1])

# input placeholder
x = tf.placeholder(tf.float32, [batch_size, im.shape[0], im.shape[1], im.shape[2]])

# Let the output size of the projective transformer be quarter of the image size.
outsize = (int(im.shape[0]/4), int(im.shape[1]/4))

# Projective Transformation Layer
stl = ProjectiveTransformer(outsize)

# Tilt the image
initial = np.array([1.5, 0.2, -0.2, 
                    0.2, 1.5, 0.0, 
                    -0.3, 0.3]).astype('float32')
initial = np.reshape(initial, [1, stl.param_dim])


# %% Run session
with tf.Session() as sess:
  with tf.device("/cpu:0"):
    with tf.variable_scope('spatial_transformer') as scope:
        # Random jitter of the parameters
        theta = initial + 0.05*tf.random_normal([batch_size, stl.param_dim])
        result = stl.transform(x, theta)

    # %% Run session
    sess.run(tf.global_variables_initializer())
    result_ = sess.run(result, feed_dict={x: batch})

# save our result
for i in range(result_.shape[0]):
  scipy.misc.imsave('projective' + str(i) + '.png', result_[i])

