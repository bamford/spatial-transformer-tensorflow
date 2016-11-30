from scipy import ndimage
import tensorflow as tf
from spatial_transformer import AffineTransformer
import numpy as np
import matplotlib.pyplot as plt

def np_sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# example input image
# for details please see 
# https://github.com/tensorflow/models/issues/193

im = np.array([-1.2053933, -1.1743802, -0.75044346, -0.74455976, -1.0506268,
 -0.91364104, -0.21054152, 0.1543106, 0.032554384, -0.52717745,
-0.66026419, -0.021319218, -0.060581781, -0.099243492, -0.26127103,
 -0.52252597, 0.1389422, -0.13638327, 0.033274196, -0.20344208,
  -0.53625256, 0.02523746, -0.076311894, 0.10775769, 0.20])

im = im.reshape(5, 5, 1)
im = im.astype('float32')

# %% Let the output size of the transformer be 5 times the image size.
out_size = (30, 30)
stl = AffineTransformer(out_size)

# %% Simulate batch
num_batch = 4
batch = np.expand_dims(im, axis=0)
batch = np.tile(batch, [num_batch, 1, 1, 1])

x = tf.placeholder(tf.float32, [num_batch, 5, 5, 1])

# Identity transform
initial = np.array([[1.0, 0, 0.0], [0, 1.0, 0.0]]).astype('float32')
initial = initial.flatten()

# %% Run session
with tf.Session() as sess:
  with tf.device("/cpu:0"):
    with tf.variable_scope('spatial_transformer'):
        theta = initial + tf.zeros([num_batch, stl.param_dim])
        result = stl.transform(x, theta)
        result = tf.sigmoid(result)

    sess.run(tf.initialize_all_variables())
    result_ = sess.run(result, feed_dict={x: batch})

# save our result
imgplot = plt.imshow(result_[0].reshape(30,30))
imgplot.set_cmap('gray')
plt.savefig("interp_stn.png")

# save ground truth result of ndimage. "order=1" uses bilinear interpolation
imgplot = plt.imshow(np_sigmoid(ndimage.interpolation.zoom(im.reshape(5,5),6.0, order=1)))
imgplot.set_cmap('gray')
plt.savefig("interp_scipy.png")

