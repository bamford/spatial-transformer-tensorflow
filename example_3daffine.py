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
pad_size = 12
vol = np.pad(vol, pad_width=[[pad_size,pad_size], [pad_size,pad_size], [pad_size,pad_size]], mode='constant')
model.dims = (np.array(model.dims) + 2*pad_size).tolist()

# input batch
batch_size = 5
batch = np.expand_dims(vol, axis=3)
batch = np.expand_dims(batch, axis=0)
batch = np.tile(batch, [batch_size, 1, 1, 1, 1])

# input placeholder
# depth, height, width, in_channels
x = tf.placeholder(tf.float32, [batch_size, vol.shape[0], vol.shape[1], vol.shape[2], 1])

outsize = (int(vol.shape[0]), int(vol.shape[1]), int(vol.shape[2]))

# Affine Transformation Layer
stl = AffineVolumeTransformer(outsize)
theta = tf.placeholder(tf.float32, [batch_size, stl.param_dim])

# Identity transformation parameters
initial = np.array([1.0, 0.0, 0.0, 0.0,
                    0.0, 1.0, 0.0, 0.0,
                    0.0, 0.0, 1.0, 0.0 ]).astype('float32')
initial = np.reshape(initial, [1, stl.param_dim])

# x-axis-rot, y-axis-rot, z-axis-rot
def transmat(phi, theta, psi, shiftmat=None):
    batch_size = phi.shape[0]
    assert batch_size==theta.shape[0] and batch_size==psi.shape[0], 'must have same number of angles for x,y and z axii'
    assert phi.ndim==1 and theta.ndim==1 and psi.ndim==1, 'must be 1 dimensional array'

    if shiftmat is None:
        shiftmat = np.zeros([batch_size,3,1])

    rotmat = np.zeros([batch_size, 3,3])
    rotmat[:,0,0] = np.cos(theta)*np.cos(psi)
    rotmat[:,0,1] = np.cos(phi)*np.sin(psi) + np.sin(phi)*np.sin(theta)*np.cos(psi)
    rotmat[:,0,2] = np.sin(phi)*np.sin(psi) - np.cos(phi)*np.sin(theta)*np.cos(psi)
    rotmat[:,1,0] = -np.cos(theta)*np.sin(psi)
    rotmat[:,1,1] = np.cos(phi)*np.cos(psi) - np.sin(phi)*np.sin(theta)*np.sin(psi)
    rotmat[:,1,2] = np.sin(phi)*np.cos(psi) + np.cos(phi)*np.sin(theta)*np.sin(psi)
    rotmat[:,2,0] = np.sin(theta)
    rotmat[:,2,1] = -np.sin(phi)*np.cos(theta)
    rotmat[:,2,2] = np.cos(phi)*np.cos(theta)

    transmat = np.concatenate([rotmat, shiftmat],2)
    return np.reshape(transmat, [batch_size, -1]).astype(np.float32)

def get_rot_diff(transmat_src, transmat_trg):
    transmat_src = np.reshape(transmat_src, [3,4])
    transmat_trg = np.reshape(transmat_trg, [3,4])
    R_src = transmat_src[:,0:3]
    t_src = transmat_src[:,3:4]
    R_trg = transmat_trg[:,0:3]
    t_trg = transmat_trg[:,3:4]
    R_res = np.dot(R_src.T, R_trg)
    t_res = -t_src + t_trg
    result = np.concatenate([R_res, t_res], 1)
    return result.flatten()


cur_angles = 2*np.pi*(2*(np.random.rand(1, 3)-0.5))
cur_theta1 = transmat(cur_angles[:,0], cur_angles[:,1], cur_angles[:,2])
print(cur_theta1.shape)
# Run session
with tf.Session(config=tf.ConfigProto(device_count={'GPU':0})) as sess:
    with tf.device("/cpu:0"):
        with tf.variable_scope('spatial_transformer') as scope:
            theta_canonical = np.tile(cur_theta1, [batch_size, 1])

            random_angles = 2*np.pi*(2*(np.random.rand(batch_size,3)-0.5))
            theta_random = transmat(random_angles[:,0], random_angles[:,1], random_angles[:,2])
            print(theta_random.shape)

            theta_inverse = np.zeros([batch_size, stl.param_dim], dtype=np.float32)
            for i in xrange(batch_size):
                cur_theta = get_rot_diff(theta_random[i,:], theta_canonical[i,:])
                theta_inverse[i,:] = cur_theta

            transformed = stl.transform(x, theta)

        sess.run(tf.global_variables_initializer())
        batch = np.transpose(batch, [0,3,2,1,4])

        x_canonical = sess.run(transformed, feed_dict={x: batch, theta: theta_canonical})
        x_random = sess.run(transformed, feed_dict={x: batch, theta: theta_random})
        x_inverse = sess.run(transformed, feed_dict={x: x_random, theta: theta_inverse})

        x_canonical = np.transpose(x_canonical, [0,3,2,1,4])
        x_random = np.transpose(x_random, [0,3,2,1,4])
        x_inverse = np.transpose(x_inverse, [0,3,2,1,4])

# save our result
def save_binvox(cur_vol, name):
    cur_model = binvox_rw.Voxels(
            data=cur_vol, 
            dims=model.dims, 
            translate=model.translate, 
            scale=model.scale, 
            axis_order=model.axis_order)
    with open(name + '.binvox', 'wb') as f:
        cur_model.write(f)

cur_vol = x_canonical[0,:,:,:,0]>0.5 # binary
save_binvox(cur_vol, 'model_canonical')

for i in range(batch_size):
    cur_vol = x_random[i,:,:,:,0]>0.5 # binary
    save_binvox(cur_vol, 'model_' + str(i) + 'random')
    cur_vol = x_inverse[i,:,:,:,0]>0.5 # binary
    save_binvox(cur_vol, 'model_' + str(i) + 'inverse')


