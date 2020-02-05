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

# THE BACKWARD FUNCTIONALITY IS NOT CURRENTLY IMPLEMENTED IN
# ELASTICTRANSFORMER, SO THIS TEST DOES NOT WORK!

import imageio
import tensorflow.compat.v1 as tf
from spatial_transformer import ElasticTransformer
import numpy as np

# Input image retrieved from:
# https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
im = imageio.imread('data/cat.jpg')
#im = imageio.imread('data/cat_small.png')
im = im / 255.
im = im.astype('float32')

# input batch
batch_size = 4
batch = np.expand_dims(im, axis=0)
batch = np.tile(batch, [batch_size, 1, 1, 1])

# Let the output size of the projective transformer be quarter of the image size.
outsize = (int(im.shape[0]), int(im.shape[1]))

# Elastic Transformation Layer
stl = ElasticTransformer(outsize)

# Random jitter of identity parameters
cur_theta = 0.1*np.random.randn(batch_size, stl.param_dim)
#cur_theta = 0.0 + 0.2*np.random.randn(1, stl.param_dim)
#cur_theta = np.repeat(cur_theta, batch_size, axis=0)
side_dim = np.floor(np.sqrt(stl.param_dim//2))
for i in range(stl.param_dim//2):
    if i<side_dim or i>=(side_dim-1)*side_dim or i%side_dim==0 or i%side_dim==side_dim-1:
        #print(i)
        cur_theta[:,i] = 0.0
        cur_theta[:,stl.param_dim//2 + i] = 0.0

# Run session
def forward(x, theta):
    return stl.transform(x, theta)

def backward(x, theta):
    return stl.transform(x, theta, False)

# %% Run session
result_ = forward(batch, cur_theta)
result__ = backward(result_, cur_theta)

# save our result_
for i in range(result_.shape[0]):
  imageio.imsave('elastic' + str(i) + '.png', result_[i])

# save our result__
for i in range(result__.shape[0]):
  imageio.imsave('elasticr' + str(i) + '.png', result__[i])
