# Spatial Transformer Network with Affine, Projective and Elastic Transformations

The Spatial Transformer Network [1] allows the spatial manipulation of data within the network.

<div align="center">
  <img width="600px" src="imgs/teaser.png"><br>
</div>

A Spatial Transformer Network implemented in Tensorflow 0.9 by Daniyar Turmukhambetov (@dkantz) [2] and based on [3] \(which is also in [4]\), [5] and [6].

This tensorflow implementation supports Affine, Projective and Elastic (Thin Plate Spline [7]) Transformations.

The original code has been updated in a number of ways by Steven Bamford (@bamford):

* now compatibile with TensorFlow's eager execution model
* transformation classes now inherit from Keras Layer
* include a Restricted Transformation (see below for explanation)
* optional edge masking with constant value, rather than nearest neighbour.

<div align="center">
  <img src="imgs/pipeline.png"><br>
</div>

## How to use

```python
from spatial_transformer import AffineTransformer, ProjectiveTransformer, ElasticTransformer, RestrictedTransformer

# Initialize
outsize = [300, 300]
stl1 = AffineTransformer(outsize)
stl2 = ProjectiveTransformer(outsize)
stl3 = ElasticTransformer(outsize)
stl4 = RestrictedTransformer(outsize)

# Transform 
y1 = stl1.transform(U, theta1)
y2 = stl2.transform(U, theta2)
y3 = stl3.transform(U, theta3)
y4 = stl4.transform(U, theta4)
```


## Examples 
### Input 
<div align="center">
  <img src="imgs/src.png">
</div>

### AffineTransformer
example_affine.py shows how to use AffineTransformer. Note, affine transformations preserve parallel lines.
<div align="center">
  <img src="imgs/affine0.png">
  <img src="imgs/affine1.png">
  <img src="imgs/affine2.png">
  <img src="imgs/affine3.png">
</div>

### ProjectiveTransformer
example_project.py shows how to use ProjectiveTransformer. Note, parallel lines are not parallel after transformation.
<div align="center">
  <img src="imgs/projective0.png">
  <img src="imgs/projective1.png">
  <img src="imgs/projective2.png">
  <img src="imgs/projective3.png">
</div>

### ElasticTransformer
example_elastic.py shows how to use ElasticTransformer. Here, deformations are defined with Thin Plate Splines on a 4x4 grid of control points.
<div align="center">
  <img src="imgs/elastic0.png">
  <img src="imgs/elastic1.png">
  <img src="imgs/elastic2.png">
  <img src="imgs/elastic3.png">
</div>

### RestrictedTransformer
example_affine.py shows how to use RestrictedTransformer. This behaves similarly to AffineTransformer, but takes more directly comprehensible parameters, designed in such a way as to restrict possible transformations. The transformation parameters, `theta`, are (`x_scale`, `y_scale`, `rotation`, `x_translation`, `y_translation`), where the scales are logarithms of the actual scale factor and the rotation is given as tan(angle/2). This prevents reflections and limits rotations to ±180 degrees. The parameterisation also makes it easier to apply further external restrictions on the transformations (e.g. not allowing rotations, only allowing isotropic scaling, etc.). 

<div align="center">
  <img src="imgs/restricted0.png">
  <img src="imgs/restricted1.png">
  <img src="imgs/restricted2.png">
  <img src="imgs/restricted3.png">
</div>

### Bilinear and Bicubic Interpolation
example_interp.py shows how to use Bilinear and Bicubic interpolation methods.

Bilinear:
<div align="center">
  Spatial Transformer Output:<br />
  <img src="imgs/interp_bilinear_stn.png"><br />
  Tensorflow Output:<br />
  <img src="imgs/interp_bilinear_tf.png"><br />
  Normalized absolute difference:<br />
  <img src="imgs/interp_diff_bilinear.png">
</div>

Bicubic:
<div align="center">
  Spatial Transformer Output:<br />
  <img src="imgs/interp_bicubic_stn.png"><br />
  Tensorflow Output:<br />
  <img src="imgs/interp_bicubic_tf.png"><br />
  Normalized absolute difference:<br />
  <img src="imgs/interp_diff_bicubic.png">
</div>

Also, the interpolation doesn't have the bug at the edges, as in [2] and [3]. See https://github.com/tensorflow/models/issues/193 for details.


## References

[1] Jaderberg, Max, et al. "Spatial Transformer Networks." 
    arXiv preprint arXiv:1506.02025 (2015)

[2] https://github.com/dantkz/spatial-transformer-tensorflow

[3] https://github.com/tensorflow/models/tree/master/transformer/transformerlayer.py

[4] https://github.com/daviddao/spatial-transformer-tensorflow

[5] https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

[6] https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py

[7] Fred L. Bookstein. "Principal warps: thin-plate splines and the decomposition of deformations."
    IEEE Transactions on Pattern Analysis and Machine Intelligence. (1989)
    http://doi.org/10.1109/34.24792

