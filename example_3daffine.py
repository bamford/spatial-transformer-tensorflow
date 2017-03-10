from scipy import ndimage 
import tensorflow as tf
from spatial_transformer import AffineVolumeTransformer
import numpy as np
import scipy.misc
import binvox_rw

# Input image retrieved from:
# https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
with open('data/model.binvox', 'rb') as f:
    model = binvox_rw.read_as_3d_array(f)

vol = model.data.copy().astype(np.float32)

# input batch
batch_size = 12
batch = np.expand_dims(vol, axis=3)
batch = np.expand_dims(batch, axis=0)
batch = np.tile(batch, [batch_size, 1, 1, 1, 1])

# input placeholder
# depth, height, width, in_channels
x = tf.placeholder(tf.float32, [batch_size, vol.shape[0], vol.shape[1], vol.shape[2], 1])

outsize = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))

# Affine Transformation Layer
stl = AffineVolumeTransformer(outsize)

# Identity transformation parameters
initial = np.array([1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0 ]).astype('float32')
initial = np.reshape(initial, [1, stl.param_dim])

# x-axis-rot, y-axis-rot, z-axis-rot
def transmat(phi, theta, psi, shiftmat=None):
    if shiftmat is None:
        shiftmat = np.zeros([3,1])

    rotmat = np.zeros([3,3])
    rotmat[0,0] = np.cos(theta)*np.cos(psi)
    rotmat[0,1] = np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi)
    rotmat[0,2] = np.sin(phi)*np.sin(psi) - np.cos(phi)*np.sin(theta)*np.cos(psi)
    rotmat[1,0] = -np.cos(theta)*np.sin(psi)
    rotmat[1,1] = np.cos(phi)*np.cos(psi) - np.sin(phi)*np.sin(theta)*np.sin(psi)
    rotmat[1,2] = np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)
    rotmat[2,0] = np.sin(theta)
    rotmat[2,1] = -np.sin(phi)*np.cos(theta)
    rotmat[2,2] = np.cos(phi)*np.cos(theta)

    if shiftmat.ndim==1:
        shiftmat = np.expand_dims(shiftmat, axis=1)
    transmat = np.concatenate([rotmat, shiftmat],1)
    print(transmat)
    return transmat.flatten().astype(np.float32)

# Run session
with tf.Session() as sess:
    with tf.device("/cpu:0"):
        with tf.variable_scope('spatial_transformer') as scope:
            # Random jitter of the identity parameters
            theta_ = np.zeros([batch_size, stl.param_dim], dtype=np.float32)
            angle_step = 2*np.pi/batch_size
            for i in xrange(batch_size):
                theta_[i,:] = transmat(0,0,i*angle_step)
            theta = tf.convert_to_tensor(theta_)
            #theta = initial + 0.1*tf.random_normal([batch_size, stl.param_dim])
            result = stl.transform(x, theta)

        sess.run(tf.global_variables_initializer())
        result_ = sess.run(result, feed_dict={x: batch})

# save our result
for i in range(result_.shape[0]):
    cur_vol = result_[i,:,:,:,0]>0.5 # binary
    cur_model = binvox_rw.Voxels(
            data=cur_vol, 
            dims=model.dims, 
            translate=model.translate, 
            scale=model.scale, 
            axis_order=model.axis_order)
    with open('affine3d' + str(i) + '.binvox', 'wb') as f:
        cur_model.write(f)


