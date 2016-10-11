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
num_batch = 5
batch = im
for i in range(num_batch-1):
  batch = np.append(batch, im, axis=0)

x = tf.placeholder(tf.float32, [num_batch, 1200, 1600, 3])
x = tf.cast(batch, 'float32')

# %% Run session
with tf.Session() as sess:
  with tf.device("/cpu:0"):
    # %% Create localisation network and convolutional layer
    stl = AffineTransformer()
    with tf.variable_scope('spatial_transformer_0'):
    
        # %% Create a fully-connected layer with 6 output nodes
        n_fc = 6
        W_fc1 = tf.Variable(tf.zeros([1200*1600*3, n_fc]), name='W_fc1')
    
        # %% Zoom into the image
        initial = np.array([[1.0, 0, 0.0], [0, 1.0, 0.0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
    
        b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        h_fc1 = tf.matmul(tf.zeros([num_batch, 1200*1600*3]), W_fc1) + b_fc1
        h_trans = stl.transform(x, h_fc1, out_size)

    # %% Run session
    sess.run(tf.initialize_all_variables())
    y = sess.run(h_trans, feed_dict={x: batch})

# save our result
scipy.misc.imsave('example1_stn.png', y[0])

