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
import imageio
import tensorflow.compat.v1 as tf
from spatial_transformer import RestrictedTransformer
import numpy as np
import imageio

# Input image retrieved from:
# https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
im = imageio.imread("data/cat.jpg")
im = im / 255.0
im = im.astype("float32")

# input batch
batch_size = 4
batch = np.expand_dims(im, axis=0)
batch = np.tile(batch, [batch_size, 1, 1, 1])

# Let the output size of the affine transformer be quarter of the image size.
outsize = (int(im.shape[0] / 4), int(im.shape[1] / 4))

# Restricted Transformation Layer
stl = RestrictedTransformer(outsize)

# Identity transformation parameters
initial = np.array([0.0, 0.0, 0.0, 0.0, 0.0]).astype("float32")
initial = np.reshape(initial, [1, stl.param_dim])

# Run session
def main(x):
    # Random jitter of the identity parameters
    theta = initial + 0.1 * tf.random_normal([batch_size, stl.param_dim])
    result = stl.transform(x, theta)
    return result


result_ = main(batch)

# save our result
for i in range(result_.shape[0]):
    imageio.imsave("restricted" + str(i) + ".png", result_[i])
