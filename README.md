# Spatial Transformer Network with Affine and Thin Plate Spline Transformations

The Spatial Transformer Network [1] allows the spatial manipulation of data within the network.

<div align="center">
  <img width="600px" src="http://i.imgur.com/ExGDVul.png"><br><br>
</div>

A Spatial Transformer Network implemented in Tensorflow 0.9 and based on [2] \(which is also in [3]\), and [4].

This tensorflow implementation supports Affine Transformations and Thin Plate Spline Deformations [5].


<div align="center">
  <img src="http://i.imgur.com/gfqLV3f.png"><br><br>
</div>

## How to use

### Affine Transformation

```python
from stn import AffineTransformer

# Initialize
out_size = [300, 300]
stl1 = AffineTransformer(out_size)

# Transform 
stl1.transform(U, theta)
```

#### Parameters
    out_size : tuple of two ints
        The size of the output of the network

    U : float 
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels]. 

    theta : float   
        The output of the localisation network,
        should have size [num_batch, 6].

To initialize the network to the identity transform init ``theta`` to :
```python
identity = np.array([[1., 0., 0.],
                    [0., 1., 0.]]) 
identity = identity.flatten()
theta = tf.Variable(initial_value=identity)
```        

### Thin Plate Splines STN

```python
from stn import TPSTransformer

# Initialize
out_size = [300, 300]
num_control_points = 16 
stl2 = TPSTransformer(out_size, num_control_points)

# Transform 
stl2.transform(U, theta)
```

#### Parameters
    out_size : tuple of two ints
        The size of the output of the network

    num_control_points : int
        The number of control points that define 
        Thin Plate Splines deformation field. 
        *MUST* be a square of an integer. 
        16 by default.

    U : float 
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels]. 

    theta : float   
        The output of the localisation network,
        should have size [num_batch, num_control_points x 2].
        num_control_points must be a square of an integer.
        
To initialize the network to the identity transform init ``theta`` to zeros:
```python
identity = np.zeros(num_control_points,2).astype('float32')
theta = tf.Variable(initial_value=identity)
```        

### References

[1] Jaderberg, Max, et al. "Spatial Transformer Networks." 
    arXiv preprint arXiv:1506.02025 (2015)

[2] https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

[3] https://github.com/tensorflow/models/tree/master/transformer/transformerlayer.py

[4] https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py

[5] Fred L. Bookstein. "Principal warps: thin-plate splines and the decomposition of deformations."
    IEEE Transactions on Pattern Analysis and Machine Intelligence. (1989)
    http://doi.org/10.1109/34.24792

