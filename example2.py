from scipy import ndimage
import tensorflow as tf
from stn import AffineTransformer
import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1.0/(1.0+np.exp(-x))

# example input image
im = np.array([-1.2053933, -1.1743802, -0.75044346, -0.74455976, -1.0506268,
 -0.91364104, -0.21054152, 0.1543106, 0.032554384, -0.52717745,
-0.66026419, -0.021319218, -0.060581781, -0.099243492, -0.26127103,
 -0.52252597, 0.1389422, -0.13638327, 0.033274196, -0.20344208,
  -0.53625256, 0.02523746, -0.076311894, 0.10775769, 0.20])

im = im.reshape(1, 5, 5, 1)
im = im.astype('float32')

# %% Let the output size of the transformer be 5 times the image size.
out_size = (30, 30)

# %% Simulate batch
batch = np.append(im, im, axis=0)
batch = np.append(batch, im, axis=0)
num_batch = 3

x = tf.placeholder(tf.float32, [num_batch, 5, 5, 1])
x = tf.cast(batch, 'float32')

# %% Run session
with tf.Session() as sess:
  with tf.device("/cpu:0"):
    stl = AffineTransformer(out_size)
    # %% Create localisation network and convolutional layer
    with tf.variable_scope('spatial_transformer_0'):
    
        # %% Create a fully-connected layer with 6 output nodes
        n_fc = 6
        W_fc1 = tf.Variable(tf.zeros([5*5*1, n_fc]), name='W_fc1')
    
        # %% Zoom into the image
        initial = np.array([[1.0, 0, 0.0], [0, 1.0, 0.0]])
        initial = initial.astype('float32')
        initial = initial.flatten()
    
        b_fc1 = tf.Variable(initial_value=initial, name='b_fc1')
        h_fc1 = tf.matmul(tf.zeros([num_batch, 5*5*1]), W_fc1) + b_fc1
        h_trans = stl.transform(x, h_fc1)
        h_trans = tf.sigmoid(h_trans)

    sess.run(tf.initialize_all_variables())
    y = sess.run(h_trans, feed_dict={x: batch})

# save our result
imgplot = plt.imshow(y[0].reshape(30,30))
imgplot.set_cmap('gray')
plt.savefig("example2_stn.png")

# save ground truth result of ndimage. "order=1" uses bilinear interpolation
imgplot = plt.imshow(sigmoid(ndimage.interpolation.zoom(im.reshape(5,5),6.0, order=1)))
imgplot.set_cmap('gray')
plt.savefig("example2_scipy.png")

