import tensorflow as tf
import numpy as np

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
        The size of the output of the spatial network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

    Notes
    -----
    To initialize the network to the identity transform initialize ``theta`` to :
        identity = np.array([[1., 0., 0.],
                             [0., 1., 0.]])
        identity = identity.flatten()
        theta = tf.Variable(initial_value=identity)

    """
    with tf.variable_scope(name):
        output, tmp = _affine_transform(theta, U, out_size)
        return output, tmp


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
        num_batch, height, width, num_channels = input_dim.get_shape().as_list()
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
        grid = tf.reshape(grid, [num_batch, 3, -1])

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
        return output, x_s_flat


def tps_transformer(U, theta, out_size, name='SpatialTPSTransformer', **kwargs):
    """Spatial Thin Plate Spline Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and edited by Daniyar Turmukhambetov for Tensorflow.

    Parameters
    ----------
    U : float
        The output of a convolutional net should have the
        shape [num_batch, height, width, num_channels].
    theta: tensor of [num_batch, num_control_points x 2]  floats
        num_control_points: a square of an int.
        The number of control points of the thin plate spline deformation
        Theta is the output of the localisation network, so it is 
        the x and y coordinates of the destination offsets of each control point.
    out_size: tuple of two ints
        The size of the output of the spatial network (height, width)

    References
    ----------
    .. [1]  Spatial Transformer Networks
            Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
            Submitted on 5 Jun 2015
    .. [2]  https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py

    Notes
    -----
    To initialize the network to the identity transform initialize ``theta`` to zeros
    """

    with tf.variable_scope(name):
        right_mat, L_inv, source_points = _initialize_tps(U, theta, out_size)
        output, dest_points = _tps_transform(theta, U, out_size, right_mat, L_inv, source_points)
        return output, dest_points

def _tps_transform(
        dest_offsets, U, out_size, right_mat, L_inv, source_points):

    num_batch, height, width, num_channels = U.get_shape().as_list()
    num_control_points = source_points.shape[1]

    out_height = out_size[0]
    out_width = out_size[1]

    # reshape destination offsets to be (num_batch, 2, num_control_points)
    # and add to source_points
    dest_points = np.tile(source_points, [num_batch, 1, 1]) + tf.reshape(
            dest_offsets, tf.pack([num_batch, 2, num_control_points]))

    # Solve as in ref [2]
    dest_points = tf.reshape(dest_points, [num_batch*2, num_control_points])
    coefficients = tf.matmul(dest_points, tf.transpose(L_inv[:, 3:]))
    coefficients = tf.reshape(coefficients, [num_batch, 2, -1])

    # Transform each point on the source grid (image_size x image_size)
    right_mat = tf.tile(tf.expand_dims(right_mat, 0), (num_batch, 1, 1))
    transformed_points = tf.batch_matmul(coefficients, right_mat)

    # TODO this is ugly, but it works!
    transformed_points = tf.reshape(transformed_points, [num_batch, 2, out_width, out_height])
    transformed_points = tf.transpose(transformed_points, [0, 1, 3, 2])
    transformed_points = tf.reshape(transformed_points, [num_batch, 2, out_height*out_width])

    #transformed_points = tf.transpose(transformed_points, [0, 1, 2])
    y_s_flat = tf.reshape(transformed_points[:,0,:], [-1])
    x_s_flat = tf.reshape(transformed_points[:,1,:], [-1])

    input_transformed = _interpolate(
            U, x_s_flat, y_s_flat,
            out_size)
    
    #output = tf.reshape(input_transformed, 
    #        tf.pack([num_batch, num_channels, out_height, out_width]))
    #output = tf.transpose(output, [0, 2, 3, 1])
    output = tf.reshape(input_transformed, 
           tf.pack([num_batch, out_height, out_width, num_channels]))
    #output = tf.transpose(output, [0, 2, 3, 1])
    return output, x_s_flat


def _U_func_numpy(x1, y1, x2, y2):
    """
    Function which implements the U function from Bookstein paper
    :param x1: x coordinate of the first point
    :param y1: y coordinate of the first point
    :param x2: x coordinate of the second point
    :param y2: y coordinate of the second point
    :return: value of z
    """

    # Return zero if same point
    if x1 == x2 and y1 == y2:
        return 0.

    # Calculate the squared Euclidean norm (r^2)
    r_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2

    # Return the squared norm (r^2 * log r^2)
    return r_2 * np.log(r_2)



def _initialize_tps(U, theta, out_size):
    """
    Initializes the thin plate spline calculation by creating the source
    point array and the inverted L matrix used for calculating the
    transformations as in ref [2]_

    :param U: the input
    :param theta: tensor of [num_control_points x 2] elements specifying the
        x and y coordinates of deformation offsets
    :param out_size: the output size
    :return:
        right_mat: shape (num_control_points + 3, out_height*out_width) tensor
        L_inv: shape (num_control_points + 3, num_control_points + 3) tensor
        source_points: shape (2, num_control_points) tensor
    """

    # break out input_shape
    num_batch, height, width, num_channels = U.get_shape().as_list()

    #The number of control points. Must be a
    #    perfect square. Points will be used to generate an evenly spaced grid.
    theta = tf.reshape(theta, [num_batch, -1, 2])
    num_control_points = theta.get_shape().as_list()[1]

    # Create source grid
    grid_size = np.sqrt(num_control_points)
    #TODO assert num_control_points is a square of an int

    x_control_source, y_control_source = np.meshgrid(
        np.linspace(-1, 1, grid_size),
        np.linspace(-1, 1, grid_size))

    # Create 2 x num_points array of source points
    source_points = np.vstack(
            (x_control_source.flatten(), y_control_source.flatten()))

    # Convert to np.float32
    source_points = source_points.astype(np.float32)

    # Get number of equations
    num_equations = num_control_points + 3

    # Initialize L to be num_equations square matrix
    L = np.zeros((num_equations, num_equations), dtype=np.float32)

    # Create P matrix components
    L[0, 3:num_equations] = 1.
    L[1:3, 3:num_equations] = source_points
    L[3:num_equations, 0] = 1.
    L[3:num_equations, 1:3] = source_points.T

    # Loop through each pair of points and create the K matrix
    for point_1 in range(num_control_points):
        for point_2 in range(point_1, num_control_points):

            L[point_1 + 3, point_2 + 3] = _U_func_numpy(
                    source_points[0, point_1], source_points[1, point_1],
                    source_points[0, point_2], source_points[1, point_2])

            if point_1 != point_2:
                L[point_2 + 3, point_1 + 3] = L[point_1 + 3, point_2 + 3]

    # Invert
    L_inv = np.linalg.inv(L)

    # Construct grid
    x_t, y_t = np.meshgrid(np.linspace(-1, 1, out_size[0]),
                           np.linspace(-1, 1, out_size[1]))
    ones = np.ones(np.prod(x_t.shape))
    orig_grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
    orig_grid = orig_grid[0:2, :]
    orig_grid = orig_grid.astype(np.float32)

    # Construct right mat

    # First Calculate the U function for the new point and each source
    # point as in ref [2]
    # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
    # squared distance
    to_transform = orig_grid[:, :, np.newaxis].transpose(2, 0, 1)
    stacked_transform = np.tile(to_transform, (num_control_points, 1, 1))
    stacked_source_points =  source_points[:, :, np.newaxis].transpose(1, 0, 2)
    r_2 = np.sum((stacked_transform - stacked_source_points) ** 2, axis=1)

    # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
    log_r_2 = np.log(r_2)
    log_r_2[np.isinf(log_r_2)] = 0.
    distances = r_2 * log_r_2

    # Add in the coefficients for the affine translation (1, x, and y,
    # corresponding to a_1, a_x, and a_y)
    upper_array = np.ones(shape=(1, orig_grid.shape[1]),
                          dtype=np.float32)
    upper_array = np.concatenate([upper_array, orig_grid], axis=0)
    right_mat = np.concatenate([upper_array, distances], axis=0)

    ## Convert to tensors
    #right_mat = tf.Variable(right_mat, name='right_mat')

    ## Convert to tensors
    #L_inv = tf.Variable(L_inv, name='L_inv')
    #source_points = tf.Variable(source_points, name='source_points')

    return right_mat, L_inv, source_points
