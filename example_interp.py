from scipy import ndimage
import tensorflow as tf
from spatial_transformer import AffineTransformer
import numpy as np
import scipy.misc

# Input image retrieved from:
# https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
im = ndimage.imread('data/cat.jpg')
im = im / 255.
im = im.astype('float32')

# input batch
batch_size = 1
batch = np.expand_dims(im, axis=0)
batch = np.tile(batch, [batch_size, 1, 1, 1])

# input placeholder
x = tf.placeholder(tf.float32, [batch_size, im.shape[0], im.shape[1], im.shape[2]])

# Let the output size of the affine transformer be quarter of the image size.
outsize = (int(im.shape[0]/4), int(im.shape[1]/4))
stl_bilinear = AffineTransformer(outsize, interp_method='bilinear')
stl_bicubic = AffineTransformer(outsize, interp_method='bicubic')

# Identity transform
initial = np.array([[1.0, 0, 0.0], [0, 1.0, 0.0]]).astype('float32')
initial = initial.flatten()

# %% Run session
with tf.Session() as sess:
  with tf.device("/cpu:0"):
    with tf.variable_scope('spatial_transformer'):
        theta = initial + tf.zeros([batch_size, stl_bilinear.param_dim])

        result_bilinear = stl_bilinear.transform(x, theta)
        result_bicubic = stl_bicubic.transform(x, theta)
        result_bilinear_tf = tf.image.resize_bilinear(x, outsize, align_corners=True)
        result_bicubic_tf = tf.image.resize_bicubic(x, outsize, align_corners=True)
        #result_bilinear_tf = tf.image.resize_bilinear(x, outsize)
        #result_bicubic_tf = tf.image.resize_bicubic(x, outsize)

    sess.run(tf.initialize_all_variables())
    result_bilinear_ = sess.run(result_bilinear, feed_dict={x: batch})
    result_bicubic_ = sess.run(result_bicubic, feed_dict={x: batch})
    result_bilinear_tf_ = sess.run(result_bilinear_tf, feed_dict={x: batch})
    result_bicubic_tf_ = sess.run(result_bicubic_tf, feed_dict={x: batch})

# save our result
scipy.misc.imsave('interp_bilinear_stn.png', result_bilinear_[0])
scipy.misc.imsave('interp_bicubic_stn.png', result_bicubic_[0])

# save tf.image result
scipy.misc.imsave('interp_bilinear_tf.png', result_bilinear_tf_[0])
scipy.misc.imsave('interp_bicubic_tf.png', result_bicubic_tf_[0])

# save differences
diff_bilinear = result_bilinear_[0] - result_bilinear_tf_[0]
diff_bilinear = diff_bilinear - np.amin(diff_bilinear)
#diff_bilinear = diff_bilinear/(0.0001 + np.amax(diff_bilinear))
scipy.misc.imsave('interp_diff_bilinear.png', diff_bilinear)
diff_bicubic = result_bicubic_[0] - result_bicubic_tf_[0]
diff_bicubic = diff_bicubic - np.amin(diff_bicubic)
#diff_bicubic = diff_bicubic/(0.0001 + np.amax(diff_bicubic))
scipy.misc.imsave('interp_diff_bicubic.png', diff_bicubic)

