import tensorflow as tf
import numpy as np
import math

import math
"""
Implementation of Spatial Transformer Networks

References
----------
[1] Spatial Transformer Networks
    Max Jaderberg, Karen Simonyan, Andrew Zisserman, Koray Kavukcuoglu
    Submitted on 5 Jun 2015

[2] https://github.com/tensorflow/models/tree/master/transformer/transformerlayer.py

[3] https://github.com/daviddao/spatial-transformer-tensorflow

[4] https://github.com/skaae/transformer_network/blob/master/transformerlayer.py

[5] https://github.com/Lasagne/Lasagne/blob/master/lasagne/layers/special.py

[6]  Fred L. Bookstein (1989):
     Principal warps: thin-plate splines and the decomposition of deformations. 
     IEEE Transactions on Pattern Analysis and Machine Intelligence.
     http://doi.org/10.1109/34.24792

"""

class AffineTransformer(object):
    """Spatial Affine Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and [3]_. Edited by Daniyar Turmukhambetov.

    """

    def __init__(self, out_size, name='SpatialAffineTransformer', **kwargs):
        """
        Parameters
        ----------
        out_size : tuple of two ints
            The size of the output of the spatial network (height, width).
        name : string
            The scope name of the variables in this network.

        """
        self.name = name
        self.out_size = out_size
        self.grid = self._meshgrid()
        
    
    def transform(self, U, theta):
        """
        Affine Transformation of input tensor U with parameters theta

        Parameters
        ----------
        U : float
            The input tensor should have the shape 
            [batch_size, height, width, num_channels].
        theta: float
            The output of the localisation network
            should have the shape
            [batch_size, 6].
        Notes
        -----
        To initialize the network to the identity transform initialize ``theta`` to :
            identity = np.array([[1., 0., 0.],
                                 [0., 1., 0.]])
            identity = identity.flatten()
            theta = tf.Variable(initial_value=identity)

        """
        with tf.variable_scope(self.name):
            output = self._affine_transform(theta, U)

        return output


    def _meshgrid(self):
        """
        the regular grid of coordinates to sample the values after the transformation
        
        """
        with tf.variable_scope(self.name + '_meshgrid'):
            # This should be equivalent to:
            #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
            #                         np.linspace(-1, 1, height))
            #  ones = np.ones(np.prod(x_t.shape))
            #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])
            x_t = tf.matmul(tf.ones(shape=tf.pack([self.out_size[0], 1])),
                            tf.transpose(tf.expand_dims(tf.linspace(-1.0, 1.0, self.out_size[1]), 1), [1, 0]))
            y_t = tf.matmul(tf.expand_dims(tf.linspace(-1.0, 1.0, self.out_size[0]), 1),
                            tf.ones(shape=tf.pack([1, self.out_size[1]])))
    
            x_t_flat = tf.reshape(x_t, (1, -1))
            y_t_flat = tf.reshape(y_t, (1, -1))
    
            ones = tf.ones_like(x_t_flat)
            grid = tf.concat(0, [x_t_flat, y_t_flat, ones])

            # Tiling for batches
            grid = tf.expand_dims(grid, 0)
            grid = tf.reshape(grid, [-1])
            return grid
    
    def _affine_transform(self, theta, U):
        with tf.variable_scope(self.name + '_affine_transform'):
            batch_size, _, _, num_channels = U.get_shape().as_list()
            theta = tf.reshape(theta, (-1, 2, 3))

            grid = tf.tile(self.grid, tf.pack([batch_size]))
            grid = tf.reshape(grid, [batch_size, 3, -1])
    
            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.batch_matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
    
            input_transformed = _interpolate(
                U, x_s_flat, y_s_flat,
                self.out_size)
    
            output = tf.reshape(input_transformed, [batch_size, self.out_size[0], self.out_size[1], num_channels])
            return output
    

class TPSTransformer(object):
    """Spatial Thin Plate Spline Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [4]_ and [5]_. Edited by Daniyar Turmukhambetov.

    """

    def __init__(self, out_size, num_control_points=16, name='SpatialTPSTransformer', **kwargs):
        """
        Parameters
        ----------
        out_size : tuple of two ints
            The size of the output of the spatial network (height, width).
        num_control_points : int
            The number of control points that define 
            Thin Plate Splines deformation field. 
            *MUST* be a square of an integer. 
            16 by default.
        name : string
            The scope name of the variables in this network.
            
        """
        self.name = name
        self.num_control_points = int(num_control_points)
        self.out_size = out_size
        self.right_mat, self.L_inv, self.source_points =  self._initialize_tps() 



    def transform(self, U, theta, **kwargs):
        """
        Parameters
        ----------
        U : float
            The input tensor should have the shape 
            [batch_size, height, width, num_channels].
        theta: float 
            Should have the shape of [batch_size, self.num_control_points x 2]
            Theta is the output of the localisation network, so it is 
            the x and y offsets of the destination coordinates 
            of each of the control points.
        Notes
        -----
        To initialize the network to the identity transform initialize ``theta`` to zeros:
            identity = np.zeros(16*2)
            identity = identity.flatten()
            theta = tf.Variable(initial_value=identity)

        """
        with tf.variable_scope(self.name):
            output = self._tps_transform(theta, U)
        return output

    def _tps_transform(self, theta, U):
        with tf.variable_scope(self.name + '_tps_transform'):
            batch_size = U.get_shape().as_list()[0]
            source_points = tf.tile(tf.expand_dims(self.source_points, 0), [batch_size, 1, 1])
            right_mat = tf.tile(tf.expand_dims(self.right_mat, 0), (batch_size, 1, 1))

            out_height = self.out_size[0]
            out_width = self.out_size[1]
    
            # reshape destination offsets to be (batch_size, 2, num_control_points)
            # and add to source_points
            theta = source_points + tf.reshape(theta, tf.pack([-1, 2, self.num_control_points]))
    
            # Solve as in ref [2]
            theta = tf.reshape(theta, [-1, self.num_control_points])
            #coefficients = tf.matmul(theta, tf.transpose(self.L_inv[:, 3:]))
            coefficients = tf.matmul(theta, tf.transpose(self.L_inv))
            coefficients = tf.reshape(coefficients, [-1, 2, self.num_control_points+3])
            #print(coefficients)
    
            # Transform each point on the source grid (image_size x image_size)
            transformed_points = tf.batch_matmul(coefficients, right_mat)
            transformed_points = tf.reshape(transformed_points, [-1, 2, out_height*out_width])
    
            x_s_flat = tf.reshape(transformed_points[:,0,:], [-1])
            y_s_flat = tf.reshape(transformed_points[:,1,:], [-1])
    
            input_transformed = _interpolate(
                    U, x_s_flat, y_s_flat,
                    self.out_size)
            
            output = tf.reshape(input_transformed, [batch_size, out_height, out_width, -1])
            return output


    def _initialize_tps(self):
        """
        Initializes the thin plate spline calculation by creating the source
        point array and the inverted L matrix used for calculating the
        transformations as in ref [5]_
    
        Returns
        ----------
        right_mat : float
            Tensor of shape [num_control_points + 3, out_height*out_width].
        L_inv : float
            Tensor of shape [num_control_points + 3, num_control_points]. 
        source_points : float
            Tensor of shape (2, num_control_points).

        """
    
        # Create source grid
        grid_size = math.sqrt(self.num_control_points)
        assert grid_size*grid_size == self.num_control_points, 'num_control_points must be a square of an int'
    
        # Create 2 x num_points array of source points
        x_control_source, y_control_source = np.meshgrid(
            np.linspace(-1, 1, grid_size),
            np.linspace(-1, 1, grid_size))
        source_points = np.vstack((x_control_source.flatten(), y_control_source.flatten()))
        source_points = source_points.astype(np.float32)
    
        # Get number of equations
        num_equations = self.num_control_points + 3
    
        # Initialize L to be num_equations square matrix
        L = np.zeros((num_equations, num_equations), dtype=np.float32)
    
        # Create P matrix components
        L[0, 3:num_equations] = 1.
        L[1:3, 3:num_equations] = source_points
        L[3:num_equations, 0] = 1.
        L[3:num_equations, 1:3] = source_points.T

        def _U_func_numpy(x1, y1, x2, y2):
            """
            Function which implements the U function from Bookstein paper [5]_

            Parameters
            ----------

            x1 : float
                x coordinate of the first point.
            y1 : float
                y coordinate of the first point.
            x2 : float 
                x coordinate of the second point.
            y2 : float
                y coordinate of the second point.

            Returns
            ----------
            value of z

            """
        
            # Return zero if same point
            if x1 == x2 and y1 == y2:
                return 0.
        
            # Calculate the squared Euclidean norm (r^2)
            r_2 = (x2 - x1) ** 2 + (y2 - y1) ** 2
        
            # Return the squared norm (r^2 * log r^2)
            return r_2 * np.log(r_2)
    
    
        # Loop through each pair of points and create the K matrix
        for point_1 in range(self.num_control_points):
            for point_2 in range(point_1, self.num_control_points):
                L[point_1 + 3, point_2 + 3] = _U_func_numpy(
                        source_points[0, point_1], source_points[1, point_1],
                        source_points[0, point_2], source_points[1, point_2])
    
                if point_1 != point_2:
                    L[point_2 + 3, point_1 + 3] = L[point_1 + 3, point_2 + 3]
    
        # Invert
        L_inv = np.linalg.inv(L)
    
        # Construct grid
        x_t, y_t = np.meshgrid(np.linspace(-1, 1, self.out_size[1]),
                               np.linspace(-1, 1, self.out_size[0]))
        orig_grid = np.vstack([x_t.flatten(), y_t.flatten()])
        orig_grid = orig_grid.astype(np.float32)
    
        # Construct right mat
    
        # First Calculate the U function for the new point and each source
        # point as in ref [5]_
        # The U function is simply U(r) = r^2 * log(r^2), where r^2 is the
        # squared distance
        to_transform = orig_grid[:, :, np.newaxis]
        stacked_transform = np.tile(to_transform, (1, 1, self.num_control_points))
        stacked_source_points =  source_points[:, np.newaxis, :]
    
        r_2 = np.sum((stacked_transform - stacked_source_points) ** 2, axis=0).transpose()
    
        # Take the product (r^2 * log(r^2)), being careful to avoid NaNs
        log_r_2 = np.log(r_2)
        log_r_2[np.isinf(log_r_2)] = 0.
        distances = r_2 * log_r_2
    
    
        # Add in the coefficients for the affine translation (1, x, and y,
        # corresponding to a_1, a_x, and a_y)
        upper_array = np.ones(shape=(1, orig_grid.shape[1]),
                              dtype=np.float32)
        right_mat = np.concatenate([upper_array, orig_grid, distances], axis=0)
    

        # Convert to Tensors
        with tf.variable_scope(self.name):
            right_mat = tf.convert_to_tensor(right_mat, dtype=tf.float32)
            L_inv = tf.convert_to_tensor(L_inv[:,3:], dtype=tf.float32)
            source_points = tf.convert_to_tensor(source_points, dtype=tf.float32)

        return right_mat, L_inv, source_points



"""
Common Functions

"""

def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.pack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, 'int32')
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

def _interpolate(im, x, y, out_size):
    with tf.variable_scope('_interpolate'):
        # constants
        batch_size = tf.shape(im)[0]
        height = tf.shape(im)[1]
        width = tf.shape(im)[2]
        channels = tf.shape(im)[3]

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
        x = (x + 1.0) / 2.0 * (width_f-1.0)
        y = (y + 1.0) / 2.0 * (height_f-1.0)

        # do sampling
        x0_f = tf.floor(x)
        y0_f = tf.floor(y)
        x1_f = x0_f + 1
        y1_f = y0_f + 1

        x0 = tf.cast(x0_f, 'int32')
        y0 = tf.cast(y0_f, 'int32')
        x1 = tf.cast(tf.minimum(x1_f, width_f - 1), 'int32')
        y1 = tf.cast(tf.minimum(y1_f, height_f - 1), 'int32')

        dim2 = width
        dim1 = width*height
        base = _repeat(tf.range(batch_size)*dim1, out_height*out_width)
        base_y0 = base + y0*dim2
        base_y1 = base + y1*dim2
        idx_a = base_y0 + x0
        idx_b = base_y1 + x0
        idx_c = base_y0 + x1
        idx_d = base_y1 + x1

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.pack([-1, channels]))
        Ia = tf.gather(im_flat, idx_a)
        Ib = tf.gather(im_flat, idx_b)
        Ic = tf.gather(im_flat, idx_c)
        Id = tf.gather(im_flat, idx_d)

        # and finally calculate interpolated values
        wa = tf.expand_dims(((x1_f-x) * (y1_f-y)), 1)
        wb = tf.expand_dims(((x1_f-x) * (y-y0_f)), 1)
        wc = tf.expand_dims(((x-x0_f) * (y1_f-y)), 1)
        wd = tf.expand_dims(((x-x0_f) * (y-y0_f)), 1)
        output = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])
        return output

