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
#im = ndimage.imread('data/cat_small.png')
im = im / 255.
im = np.expand_dims(im, axis=0)
im = im.astype('float32')

# %% Simulate batch
num_batch = 3
batch = im
for i in range(num_batch-1):
  batch = np.append(batch, im, axis=0)
param_dim = 6

x = tf.placeholder(tf.float32, [num_batch, im.shape[1], im.shape[2], im.shape[3]])
params = tf.placeholder(tf.float32, [num_batch, param_dim])

# %% Zoom into the image
initial = np.array([[1.0, 0, 0.0], [0, 1.0, 0.0]])
initial = initial.astype('float32')
initial = initial.flatten()

# Random jitter of the zooming parameters
my_params = np.tile(np.reshape(initial, [1, param_dim]), (num_batch, 1))
#my_params = my_params + 0.1*np.random.randn(num_batch, param_dim)

# %% Run session
with tf.Session() as sess:
  with tf.device("/cpu:0"):
    # %% Create localisation network and convolutional layer

    # %% Let the output size of the affine transformer be same as the image size.
    outsize = (im.shape[1], im.shape[2])
    stl = AffineTransformer(outsize)

    with tf.variable_scope('spatial_transformer_0'):
        h_trans = stl.transform(x, params)

    # %% Run session
    sess.run(tf.initialize_all_variables())
    y = sess.run(h_trans, feed_dict={x: batch, params: my_params})

# save our result
for i in range(num_batch):
  scipy.misc.imsave('example1_stn' + str(i) + '.png', y[i])


print(y[0].shape)
print(im.shape)
diff = y[0]-im[0,:,:,:]
print(np.amax(diff))
print(np.amin(diff))
print(np.sum(diff))
diff = diff-np.amin(diff)
diff = diff/np.amax(diff)
print(diff.shape)
print(np.amax(diff))
print(np.amin(diff))
scipy.misc.imsave('example1_diff.png', diff)

