import imageio
import tensorflow.compat.v1 as tf
from spatial_transformer import AffineVolumeTransformer
import numpy as np
import binvox_rw
import sys

def read_binvox(f):
    class Model:
        pass

    model = Model()

    line = f.readline().strip()
    if not line.startswith(b'#binvox'):
        raise IOError('Not a binvox file')

    model.dims = list(map(int, f.readline().strip().split(b' ')[1:]))
    model.translate = list(map(float, f.readline().strip().split(b' ')[1:]))
    model.scale = float(f.readline().strip().split(b' ')[1])

    _ = f.readline()
    raw_data = np.frombuffer(f.read(), dtype=np.uint8)
    values, counts = raw_data[::2], raw_data[1::2]

    # xzy (binvox) -> zyx (tensorflow)
    model.data = np.transpose(np.repeat(values, counts).astype(np.bool).reshape(model.dims), (1,2,0))

    # zxy -> zyx (should all be equal, so doesn't matter)
    model.dims = [model.dims[i] for i in [0,2,1]]
    return model

def write_binvox(model, f):
    f.write(b'#binvox 1\n')
    f.write(('dim '+' '.join(map(str, [model.dims[i] for i in [0,2,1]]))+'\n').encode())
    f.write(('translate '+' '.join(map(str, model.translate))+'\n').encode())
    f.write(('scale'+str(model.scale)+'\n').encode())
    f.write(b'data\n')

    # zyx (tensorflow) -> xzy (binvox)
    voxels = np.transpose(model.data, (2, 0, 1)).flatten()

    # run length encoding
    value = voxels[0]
    count = 0

    def dump():
        if sys.version_info[0] < 3:
            # python 2
            f.write(chr(value))
            f.write(chr(count))
        else:
            # python 3
            f.write(bytes((value,)))
            f.write(bytes((count,)))

    for curval in voxels:
        if curval==value:
            count += 1
            if count==255:
                dump()
                count = 0
        else:
            dump()
            value = curval
            count = 1
    if count > 0:
        dump()


# Input image retrieved from:
# https://raw.githubusercontent.com/skaae/transformer_network/master/cat.jpg
with open('data/model.binvox', 'rb') as f:
    model = read_binvox(f)

vol = model.data.copy().astype(np.float32)
pad_size = 12
vol = np.pad(vol, pad_width=[[pad_size,pad_size], [pad_size,pad_size], [pad_size,pad_size]], mode='constant')
model.dims = (np.array(model.dims) + 2*pad_size).tolist()

# input batch
batch_size = 3
batch = np.expand_dims(vol, axis=3)
batch = np.expand_dims(batch, axis=0)
batch = np.tile(batch, [batch_size, 1, 1, 1, 1])

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


# Run session
def main(x, theta):
    random_angles = np.pi*(2*(np.random.rand(batch_size,3)-0.5))
    shifts = (np.random.rand(batch_size,3,1)-0.5)
    theta_random = transmat(random_angles[:,0], random_angles[:,1], random_angles[:,2], shifts)
    transformed = stl.transform(x, theta)
    return transformed

x_random = main(batch, theta)


class Model:
    pass
model = Model()

for i in range(batch_size):
    cur_vol = x_random[i,:,:,:,0]>0.5 # binary

    model.dims = list(cur_vol.shape)
    model.data = cur_vol
    model.translate = [0,0,0]
    model.scale = 1.0

    #print(model.dims)
    #print(model.translate)
    #print(model.scale)
    #print(model.axis_order)

    with open('model_' + str(i) + 'random.binvox', 'wb') as f:
        write_binvox(model, f)
