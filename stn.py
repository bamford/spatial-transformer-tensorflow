import tensorflow as tf
import numpy as np

#import theano
#import theano.tensor as T

#from .. import init
#from .. import nonlinearities
#from ..utils import as_tuple, floatX
#from ..random import get_rng
#from .base import Layer, MergeLayer
#from theano.sandbox.rng_mrg import MRG_RandomStreams as RandomStreams


#    "AffineTransformerLayer",
#    "TPSTransformerLayer"

def affine_transformer(U, theta, out_size, name='SpatialAffineTransformer', **kwargs):
    """Spatial Affine Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by David Dao for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: float
        The output of the
        localisation network should be [num_batch, 6].
    out_size: tuple of two ints
        The size of the output of the network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform init
    ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """

    with tf.variable_scope(name):
        output = _affine_transform(theta, U, out_size)
        return output


def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        rep = tf.transpose(
            tf.expand_dims(tf.ones(shape=tf.pack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

def _interpolate(im, x, y, out_size):
    with tf.variable_scope('_interpolate'):
        # constants
        num_batch = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

        max_y = tf.cast(tf.shape(im)[1] - 1, 'int32')
        max_x = tf.cast(tf.shape(im)[2] - 1, 'int32')

        x = tf.cast(x, 'float32')
        y = tf.cast(y, 'float32')
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        zero = tf.zeros([], dtype='int32')


        # scale indices from [-1, 1] to [0, width/height - 1]
        x = tf.clip_by_value(x, -1, 1)
        y = tf.clip_by_value(y, -1, 1)

        x = (x + 1.0) / 2.0 * (width_f-1.001)
        y = (y + 1.0) / 2.0 * (height_f-1.001)




        # do sampling
        x0 = tf.cast(tf.floor(x), 'int32')
        x1 = x0 + 1
        y0 = tf.cast(tf.floor(y), 'int32')
        y1 = y0 + 1

        x0 = tf.clip_by_value(x0, zero, max_x)
        x1 = tf.clip_by_value(x1, zero, max_x)
        y0 = tf.clip_by_value(y0, zero, max_y)
        y1 = tf.clip_by_value(y1, zero, max_y)



        dim2 = width
        dim1 = width*height
        base = _repeat(tf.range(num_batch)*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.pack([-1, channels]))
        im_flat = tf.cast(im_flat, 'float32')
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        x0_f = tf.cast(x0, 'float32')
        x1_f = tf.cast(x1, 'float32')
        y0_f = tf.cast(y0, 'float32')
        y1_f = tf.cast(y1, 'float32')
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        return output

def _meshgrid(height, width):
    with tf.variable_scope('_meshgrid'):
        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
        x_t = tf.matmul(tf.ones(shape=tf.pack([height, 1])),
                        tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, width), 1), [1, 0]))
        y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, height), 1),
                        tf.ones(shape=tf.pack([1, width])))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(0, [x_t_flat, y_t_flat, ones])
        return grid

def _affine_transform(theta, input_dim, out_size):
    with tf.variable_scope('_affine_transform'):
        num_batch = tf.shape(input_dim)[0]
        height = tf.shape(input_dim)[1]
        width = tf.shape(input_dim)[2]
        num_channels = tf.shape(input_dim)[3]
        theta = tf.reshape(theta, (-1, 2, 3))
        theta = tf.cast(theta, 'float32')

        # grid of (x_t, y_t, 1), eq (1) in ref [1]
        height_f = tf.cast(height, 'float32')
        width_f = tf.cast(width, 'float32')
        out_height = out_size[0]
        out_width = out_size[1]
        grid = _meshgrid(out_height, out_width)
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        grid = tf.tile(grid, tf.pack([num_batch]))
        grid = tf.reshape(grid, tf.pack([num_batch, 3, -1]))

        # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
        T_g = tf.batch_matmul(theta, grid)
        x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
        y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
        x_s_flat = tf.reshape(x_s, [-1])
        y_s_flat = tf.reshape(y_s, [-1])

        input_transformed = _interpolate(
            input_dim, x_s_flat, y_s_flat,
            out_size)

        output = tf.reshape(
            input_transformed, tf.pack([num_batch, out_height, out_width, num_channels]))
        return output



#class TPSTransformerLayer(MergeLayer):
#    """
#    Spatial transformer layer
#
#    The layer applies a thin plate spline transformation [2]_ on the input
#    as in [1]_. The thin plate spline transform is determined based on the
#    movement of some number of control points. The starting positions for
#    these control points are fixed. The output is interpolated with a
#    bilinear transformation.
#
#    Parameters
#    ----------
#    incoming : a :class:`Layer` instance or a tuple
#        The layer feeding into this layer, or the expected input shape. The
#        output of this layer should be a 4D tensor, with shape
#        ``(batch_size, num_input_channels, input_rows, input_columns)``.
#
#    localization_network : a :class:`Layer` instance
#        The network that calculates the parameters of the thin plate spline
#        transformation as the x and y coordinates of the destination offsets of
#        each control point. The output of the localization network  should
#        be a 2D tensor, with shape ``(batch_size, 2 * num_control_points)``
#
#    downsample_factor : float or iterable of float
#        A float or a 2-element tuple specifying the downsample factor for the
#        output image (in both spatial dimensions). A value of 1 will keep the
#        original size of the input. Values larger than 1 will downsample the
#        input. Values below 1 will upsample the input.
#
#    control_points : integer
#        The number of control points to be used for the thin plate spline
#        transformation. These points will be arranged as a grid along the
#        image, so the value must be a perfect square. Default is 16.
#
#    precompute_grid : 'auto' or boolean
#        Flag to precompute the U function [2]_ for the grid and source
#        points. If 'auto', will be set to true as long as the input height
#        and width are specified. If true, the U function is computed when the
#        layer is constructed for a fixed input shape. If false, grid will be
#        computed as part of the Theano computational graph, which is
#        substantially slower as this computation scales with
#        num_pixels*num_control_points. Default is 'auto'.
#
#    References
#    ----------
#    .. [1]  Max Jaderberg, Karen Simonyan, Andrew Zisserman,
#            Koray Kavukcuoglu (2015):
#            Spatial Transformer Networks. NIPS 2015,
#            http://papers.nips.cc/paper/5854-spatial-transformer-networks.pdf
#    .. [2]  Fred L. Bookstein (1989):
#            Principal warps: thin-plate splines and the decomposition of
#            deformations. IEEE Transactions on
#            Pattern Analysis and Machine Intelligence.
#            http://doi.org/10.1109/34.24792
#
#    Examples
#    --------
#    Here, we'll implement an identity transform using a thin plate spline
#    transform. First we'll create the destination control point offsets. To
#    make everything invariant to the shape of the image, the x and y range
#    of the image is normalized to [-1, 1] as in ref [1]_. To replicate an
#    identity transform, we'll set the bias to have all offsets be 0. More
#    complicated transformations can easily be implemented using different x
#    and y offsets (importantly, each control point can have it's own pair of
#    offsets).
#
#    >>> import numpy as np
#    >>> import lasagne
#    >>>
#    >>> # Create the network
#    >>> # we'll initialize the weights and biases to zero, so it starts
#    >>> # as the identity transform (all control point offsets are zero)
#    >>> W = b = lasagne.init.Constant(0.0)
#    >>>
#    >>> # Set the number of points
#    >>> num_points = 16
#    >>>
#    >>> l_in = lasagne.layers.InputLayer((None, 3, 28, 28))
#    >>> l_loc = lasagne.layers.DenseLayer(l_in, num_units=2*num_points,
#    ...                                   W=W, b=b, nonlinearity=None)
#    >>> l_trans = lasagne.layers.TPSTransformerLayer(l_in, l_loc,
#    ...                                          control_points=num_points)
#    """
#
#    def __init__(self, incoming, localization_network, downsample_factor=1,
#                 control_points=16, precompute_grid='auto', **kwargs):
#        super(TPSTransformerLayer, self).__init__(
#                [incoming, localization_network], **kwargs)
#
#        self.downsample_factor = as_tuple(downsample_factor, 2)
#        self.control_points = control_points
#
#        input_shp, loc_shp = self.input_shapes
#
#        # Error checking
#        if loc_shp[-1] != 2 * control_points or len(loc_shp) != 2:
#            raise ValueError("The localization network must have "
#                             "output shape: (batch_size, "
#                             "2*control_points)")
#
#        if round(np.sqrt(control_points)) != np.sqrt(
#                control_points):
#            raise ValueError("The number of control points must be"
#                             " a perfect square.")
#
#        if len(input_shp) != 4:
#            raise ValueError("The input network must have a 4-dimensional "
#                             "output shape: (batch_size, num_input_channels, "
#                             "input_rows, input_columns)")
#
#        # Process precompute grid
#        can_precompute_grid = all(s is not None for s in input_shp[2:])
#        if precompute_grid == 'auto':
#            precompute_grid = can_precompute_grid
#        elif precompute_grid and not can_precompute_grid:
#            raise ValueError("Grid can only be precomputed if the input "
#                             "height and width are pre-specified.")
#        self.precompute_grid = precompute_grid
#
#        # Create source points and L matrix
#        self.right_mat, self.L_inv, self.source_points, self.out_height, \
#            self.out_width = _initialize_tps(
#                control_points, input_shp, self.downsample_factor,
#                precompute_grid)
#
#    def get_output_shape_for(self, input_shapes):
#        shape = input_shapes[0]
#        factors = self.downsample_factor
#        return (shape[:2] + tuple(None if s is None else int(s // f)
#                                  for s, f in zip(shape[2:], factors)))
#
#    def get_output_for(self, inputs, **kwargs):
#        # see eq. (1) and sec 3.1 in [1]
#        # Get input and destination control points
#        input, dest_offsets = inputs
#        return _transform_thin_plate_spline(
#                dest_offsets, input, self.right_mat, self.L_inv,
#                self.source_points, self.out_height, self.out_width,
#                self.precompute_grid, self.downsample_factor)
#
#
#def _transform_thin_plate_spline(
#        dest_offsets, input, right_mat, L_inv, source_points, out_height,
#        out_width, precompute_grid, downsample_factor):
#
#    num_batch, num_channels, height, width = input.shape
#    num_control_points = source_points.shape[1]
#
#    # reshape destination offsets to be (num_batch, 2, num_control_points)
#    # and add to source_points
#    dest_points = source_points + T.reshape(
#            dest_offsets, (num_batch, 2, num_control_points))
#
#    # Solve as in ref [2]
#    coefficients = T.dot(dest_points, L_inv[:, 3:].T)
#
#    if precompute_grid:
#
#        # Transform each point on the source grid (image_size x image_size)
#        right_mat = T.tile(right_mat.dimshuffle('x', 0, 1), (num_batch, 1, 1))
#        transformed_points = T.batched_dot(coefficients, right_mat)
#
#    else:
#
#        # Transformed grid
#        out_height = T.cast(height // downsample_factor[0], 'int64')
#        out_width = T.cast(width // downsample_factor[1], 'int64')
#        orig_grid = _meshgrid(out_height, out_width)
#        orig_grid = orig_grid[0:2, :]
#        orig_grid = T.tile(orig_grid, (num_batch, 1, 1))
#
#        # Transform each point on the source grid (image_size x image_size)
#        transformed_points = _get_transformed_points_tps(
#                orig_grid, source_points, coefficients, num_control_points,
#                num_batch)
#
#    # Get out new points
#    x_transformed = transformed_points[:, 0].flatten()
#    y_transformed = transformed_points[:, 1].flatten()
#
#    # dimshuffle input to  (bs, height, width, channels)
#    input_dim = input.dimshuffle(0, 2, 3, 1)
#    input_transformed = _interpolate(
#            input_dim, x_transformed, y_transformed,
#            out_height, out_width)
#
#    output = T.reshape(input_transformed,
#                       (num_batch, out_height, out_width, num_channels))
#    output = output.dimshuffle(0, 3, 1, 2)  # dimshuffle to conv format
#    return output
#
#
#def _get_transformed_points_tps(new_points, source_points, coefficients,
#                                num_points, batch_size):
#    """
#    Calculates the transformed points' value using the provided coefficients
#
#    :param new_points: num_batch x 2 x num_to_transform tensor
#    :param source_points: 2 x num_points array of source points
#    :param coefficients: coefficients (should be shape (num_batch, 2,
#        control_points + 3))
#    :param num_points: the number of points
#
#    :return: the x and y coordinates of each transformed point. Shape (
#        num_batch, 2, num_to_transform)
#    """
#
#    # Calculate the U function for the new point and each source point as in
#    # ref [2]
#    # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
#    # squared distance
#
#    # Calculate the squared dist between the new point and the source points
#    to_transform = new_points.dimshuffle(0, 'x', 1, 2)
#    stacked_transform = T.tile(to_transform, (1, num_points, 1, 1))
#    r_2 = T.sum(((stacked_transform - source_points.dimshuffle(
#            'x', 1, 0, 'x')) ** 2), axis=2)
#
#    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
#    log_r_2 = T.log(r_2)
#    distances = T.switch(T.isnan(log_r_2), r_2 * log_r_2, 0.)
#
#    # Add in the coefficients for the affine translation (1, x, and y,
#    # corresponding to a_1, a_x, and a_y)
#    upper_array = T.concatenate([T.ones((batch_size, 1, new_points.shape[2]),
#                                        dtype=theano.config.floatX),
#                                 new_points], axis=1)
#    right_mat = T.concatenate([upper_array, distances], axis=1)
#
#    # Calculate the new value as the dot product
#    new_value = T.batched_dot(coefficients, right_mat)
#    return new_value
#
#
#def _U_func_numpy(x1, y1, x2, y2):
#    """
#    Function which implements the U function from Bookstein paper
#    :param x1: x coordinate of the first point
#    :param y1: y coordinate of the first point
#    :param x2: x coordinate of the second point
#    :param y2: y coordinate of the second point
#    :return: value of z
#    """
#
#    # Return zero if same point
#    if x1 == x2 and y1 == y2:
#        return 0.
#
#    # Calculate the squared Euclidean norm (r^2)
#    r_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
#
#    # Return the squared norm (r^2 * log r^2)
#    return r_2 * np.log(r_2)
#
#
#def _initialize_tps(num_control_points, input_shape, downsample_factor,
#                    precompute_grid):
#    """
#    Initializes the thin plate spline calculation by creating the source
#    point array and the inverted L matrix used for calculating the
#    transformations as in ref [2]_
#
#    :param num_control_points: the number of control points. Must be a
#        perfect square. Points will be used to generate an evenly spaced grid.
#    :param input_shape: tuple with 4 elements specifying the input shape
#    :param downsample_factor: tuple with 2 elements specifying the
#        downsample for the height and width, respectively
#    :param precompute_grid: boolean specifying whether to precompute the
#        grid matrix
#    :return:
#        right_mat: shape (num_control_points + 3, out_height*out_width) tensor
#        L_inv: shape (num_control_points + 3, num_control_points + 3) tensor
#        source_points: shape (2, num_control_points) tensor
#        out_height: tensor constant specifying the ouptut height
#        out_width: tensor constant specifying the output width
#
#    """
#
#    # break out input_shape
#    _, _, height, width = input_shape
#
#    # Create source grid
#    grid_size = np.sqrt(num_control_points)
#    x_control_source, y_control_source = np.meshgrid(
#        np.linspace(-1, 1, grid_size),
#        np.linspace(-1, 1, grid_size))
#
#    # Create 2 x num_points array of source points
#    source_points = np.vstack(
#            (x_control_source.flatten(), y_control_source.flatten()))
#
#    # Convert to floatX
#    source_points = source_points.astype(theano.config.floatX)
#
#    # Get number of equations
#    num_equations = num_control_points + 3
#
#    # Initialize L to be num_equations square matrix
#    L = np.zeros((num_equations, num_equations), dtype=theano.config.floatX)
#
#    # Create P matrix components
#    L[0, 3:num_equations] = 1.
#    L[1:3, 3:num_equations] = source_points
#    L[3:num_equations, 0] = 1.
#    L[3:num_equations, 1:3] = source_points.T
#
#    # Loop through each pair of points and create the K matrix
#    for point_1 in range(num_control_points):
#        for point_2 in range(point_1, num_control_points):
#
#            L[point_1 + 3, point_2 + 3] = _U_func_numpy(
#                    source_points[0, point_1], source_points[1, point_1],
#                    source_points[0, point_2], source_points[1, point_2])
#
#            if point_1 != point_2:
#                L[point_2 + 3, point_1 + 3] = L[point_1 + 3, point_2 + 3]
#
#    # Invert
#    L_inv = np.linalg.inv(L)
#
#    if precompute_grid:
#        # Construct grid
#        out_height = np.array(height // downsample_factor[0]).astype('int64')
#        out_width = np.array(width // downsample_factor[1]).astype('int64')
#        x_t, y_t = np.meshgrid(np.linspace(-1, 1, out_width),
#                               np.linspace(-1, 1, out_height))
#        ones = np.ones(np.prod(x_t.shape))
#        orig_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
#        orig_grid = orig_grid[0:2, :]
#        orig_grid = orig_grid.astype(theano.config.floatX)
#
#        # Construct right mat
#
#        # First Calculate the U function for the new point and each source
#        # point as in ref [2]
#        # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
#        # squared distance
#        to_transform = orig_grid[:, :, np.newaxis].transpose(2, 0, 1)
#        stacked_transform = np.tile(to_transform, (num_control_points, 1, 1))
#        stacked_source_points = \
#            source_points[:, :, np.newaxis].transpose(1, 0, 2)
#        r_2 = np.sum((stacked_transform - stacked_source_points) ** 2, axis=1)
#
#        # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
#        log_r_2 = np.log(r_2)
#        log_r_2[np.isinf(log_r_2)] = 0.
#        distances = r_2 * log_r_2
#
#        # Add in the coefficients for the affine translation (1, x, and y,
#        # corresponding to a_1, a_x, and a_y)
#        upper_array = np.ones(shape=(1, orig_grid.shape[1]),
#                              dtype=theano.config.floatX)
#        upper_array = np.concatenate([upper_array, orig_grid], axis=0)
#        right_mat = np.concatenate([upper_array, distances], axis=0)
#
#        # Convert to tensors
#        out_height = T.as_tensor_variable(out_height)
#        out_width = T.as_tensor_variable(out_width)
#        right_mat = T.as_tensor_variable(right_mat)
#
#    else:
#        out_height = None
#        out_width = None
#        right_mat = None
#
#    # Convert to tensors
#    L_inv = T.as_tensor_variable(L_inv)
#    source_points = T.as_tensor_variable(source_points)
#
#    return right_mat, L_inv, source_points, out_height, out_width
#
