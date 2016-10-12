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
import tensorflow as tf
from stn import AffineTransformer
import numpy as np
import scipy.misc

# %% Create a batch of three images (1600 x 1200)
# %% Image retrieved from:
# %% https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
im = ndimage.imread('data/cat.jpg')
im = im / 255.
im = im.reshape(1, 1200, 1600, 3)
im = im.astype('float32')

# %% Let the output size of the affine transformer be half the image size.
out_size = (600, 800)

# %% Simulate batch
num_batch = 3
batch = im
for i in range(num_batch-1):
  batch = np.append(batch, im, axis=0)
param_dim = 6

x = tf.placeholder(tf.float32, [num_batch, 1200, 1600, 3])
params = tf.placeholder(tf.float32, [num_batch, param_dim])

# %% Zoom into the image
initial = np.array([[0.5, 0, 0.0], [0, 0.5, 0.0]])
initial = initial.astype('float32')
initial = initial.flatten()

my_params = np.tile(np.reshape(initial, [1, param_dim]), (num_batch, 1))
my_params = my_params + 0.1*np.random.randn(num_batch, param_dim)

# %% Run session
with tf.Session() as sess:
  with tf.device("/cpu:0"):
    # %% Create localisation network and convolutional layer
    stl = AffineTransformer(x.get_shape().as_list(), out_size)
    with tf.variable_scope('spatial_transformer_0'):
        h_trans = stl.transform(x, params)

    # %% Run session
    sess.run(tf.initialize_all_variables())
    y = sess.run(h_trans, feed_dict={x: batch, params: my_params})

# save our result
for i in range(num_batch):
  scipy.misc.imsave('example1_stn' + str(i) + '.png', y[i])

