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

import tensorflow as tf


"""
Legacy Function

"""
def transformer(U, theta, out_size, name='SpatialTransformer', **kwargs):
    with tf.variable_scope(name):
        stl = AffineTransformer(out_size)
        output = stl.transform(U, theta, out_size)
        return output


class AffineTransformer(object):
    """Spatial Affine Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and [3]_. Edited by Daniyar Turmukhambetov.

    """

    def __init__(self, out_size, name='SpatialAffineTransformer', interp_method='bilinear', **kwargs):
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
        self.param_dim = 6
        self.interp_method=interp_method

        with tf.variable_scope(self.name):
            self.grid = _meshgrid(self.out_size)
        
    
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
            x_s, y_s = self._transform(U, theta)
    
            output = _interpolate(
                U, x_s, y_s,
                self.out_size,
                method=self.interp_method
                )
    
            batch_size, _, _, num_channels = U.get_shape().as_list()
            output = tf.reshape(output, [batch_size, self.out_size[0], self.out_size[1], num_channels])

        return output


    
    def _transform(self, U, theta):
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
            return x_s_flat, y_s_flat
    
class ProjectiveTransformer(object):
    """Spatial Projective Transformer Layer

    Implements a spatial transformer layer as described in [1]_.
    Based on [2]_ and [3]_. Edited by Daniyar Turmukhambetov.

    """

    def __init__(self, out_size, name='SpatialProjectiveTransformer', interp_method='bilinear', **kwargs):
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
        self.param_dim = 8
        self.interp_method=interp_method

        with tf.variable_scope(self.name):
            self.grid = _meshgrid(self.out_size)
        
    
    def transform(self, U, theta):
        """
        Projective Transformation of input tensor U with parameters theta

        Parameters
        ----------
        U : float
            The input tensor should have the shape 
            [batch_size, height, width, num_channels].
        theta: float
            The output of the localisation network
            should have the shape
            [batch_size, 8].
        Notes
        -----
        To initialize the network to the identity transform initialize ``theta`` to :
            identity = np.array([1., 0., 0.,
                                [0., 1., 0.,
                                [0., 0.])
            theta = tf.Variable(initial_value=identity)

        """
        with tf.variable_scope(self.name):
            x_s, y_s = self._transform(U, theta)
    
            output = _interpolate(
                U, x_s, y_s,
                self.out_size,
                method=self.interp_method
                )
    
            batch_size, _, _, num_channels = U.get_shape().as_list()
            output = tf.reshape(output, [batch_size, self.out_size[0], self.out_size[1], num_channels])

        return output


    
    def _transform(self, U, theta):
        with tf.variable_scope(self.name + '_projective_transform'):
            batch_size, _, _, num_channels = U.get_shape().as_list()
            theta = tf.reshape(theta, (batch_size, 8))
            theta = tf.concat(1, [theta, tf.ones([batch_size, 1])])
            theta = tf.reshape(theta, (batch_size, 3, 3))

            grid = tf.tile(self.grid, tf.pack([batch_size]))
            grid = tf.reshape(grid, [batch_size, 3, -1])
    
            # Transform A x (x_t, y_t, 1)^T -> (x_s, y_s)
            T_g = tf.batch_matmul(theta, grid)
            x_s = tf.slice(T_g, [0, 0, 0], [-1, 1, -1])
            y_s = tf.slice(T_g, [0, 1, 0], [-1, 1, -1])
            z_s = tf.slice(T_g, [0, 2, 0], [-1, 1, -1])

            x_s = x_s/z_s
            y_s = y_s/z_s

            x_s_flat = tf.reshape(x_s, [-1])
            y_s_flat = tf.reshape(y_s, [-1])
            return x_s_flat, y_s_flat

class ElasticTransformer(object):
    """Spatial Elastic Transformer Layer with Thin Plate Spline deformations

    Implements a spatial transformer layer as described in [1]_.
    Based on [4]_ and [5]_. Edited by Daniyar Turmukhambetov.

    """

    def __init__(self, out_size, param_dim=2*16, name='SpatialElasticTransformer', interp_method='bilinear', **kwargs):
        """
        Parameters
        ----------
        out_size : tuple of two ints
            The size of the output of the spatial network (height, width).
        param_dim: int
            The 2 x number of control points that define 
            Thin Plate Splines deformation field. 
            number of control points *MUST* be a square of an integer. 
            2 x 16 by default.
        name : string
            The scope name of the variables in this network.
            
        """
        num_control_points = int(param_dim/2)
        assert param_dim == 2*num_control_points, 'param_dim must be 2 times a square of an integer.'

        self.name = name
        self.param_dim = param_dim
        self.interp_method=interp_method
        self.num_control_points = num_control_points

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
            x_s, y_s = self._transform(U, theta)
    
            output = _interpolate(
                U, x_s, y_s,
                self.out_size,
                method=self.interp_method
                )
    
            batch_size, _, _, num_channels = U.get_shape().as_list()
            output = tf.reshape(output, [batch_size, self.out_size[0], self.out_size[1], num_channels])
        return output


    def _transform(self, U, theta):
        with tf.variable_scope(self.name + '_elastic_transform'):
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
            coefficients = tf.matmul(theta, tf.transpose(self.L_inv))
            coefficients = tf.reshape(coefficients, [-1, 2, self.num_control_points+3])
    
            # Transform each point on the source grid (image_size x image_size)
            transformed_points = tf.batch_matmul(coefficients, right_mat)
            transformed_points = tf.reshape(transformed_points, [-1, 2, out_height*out_width])
    
            x_s_flat = tf.reshape(transformed_points[:,0,:], [-1])
            y_s_flat = tf.reshape(transformed_points[:,1,:], [-1])
            return x_s_flat, y_s_flat


    def _initialize_tps(self):
        import numpy as np
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
        grid_size = np.sqrt(self.num_control_points)
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
            x1, y1 : float
                x and y coordinates of the first point.
            x2, y2 : float 
                x and y coordinates of the second point.

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
        log_r_2[np.isinf(log_r_2)] = 0
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
def _meshgrid(out_size):
    """
    the regular grid of coordinates to sample the values after the transformation
    
    """
    with tf.variable_scope('meshgrid'):

        # This should be equivalent to:
        #  x_t, y_t = np.meshgrid(np.linspace(-1, 1, width),
        #                         np.linspace(-1, 1, height))
        #  ones = np.ones(np.prod(x_t.shape))
        #  grid = np.vstack([x_t.flatten(), y_t.flatten(), ones])

        x_t, y_t = tf.meshgrid(tf.linspace(-1.0, 1.0,  out_size[1]),
                               tf.linspace(-1.0, 1.0,  out_size[0]))

        x_t_flat = tf.reshape(x_t, (1, -1))
        y_t_flat = tf.reshape(y_t, (1, -1))

        ones = tf.ones_like(x_t_flat)
        grid = tf.concat(0, [x_t_flat, y_t_flat, ones])

        # Tiling for batches
        grid = tf.expand_dims(grid, 0)
        grid = tf.reshape(grid, [-1])
        return grid


def _repeat(x, n_repeats):
    with tf.variable_scope('_repeat'):
        rep = tf.transpose(tf.expand_dims(tf.ones(shape=tf.pack([n_repeats, ])), 1), [1, 0])
        rep = tf.cast(rep, tf.int32)
        x = tf.matmul(tf.reshape(x, (-1, 1)), rep)
        return tf.reshape(x, [-1])

def _interpolate(im, x, y, out_size, method):
    if method=='bilinear':
        return bilinear_interp(im, x, y, out_size)
    if method=='bicubic':
        return bicubic_interp(im, x, y, out_size)
    return None


def bilinear_interp(im, x, y, out_size):
    with tf.variable_scope('bilinear_interp'):
        batch_size, height, width, channels = im.get_shape().as_list()
        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)
        out_height = out_size[0]
        out_width = out_size[1]

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

        x0 = tf.cast(x0_f, tf.int32)
        y0 = tf.cast(y0_f, tf.int32)
        x1 = tf.cast(tf.minimum(x1_f, width_f - 1),  tf.int32)
        y1 = tf.cast(tf.minimum(y1_f, height_f - 1), tf.int32)

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


def bicubic_interp(im, x, y, out_size):
    alpha = -0.75 # same as in tf.image.resize_images, see:
    #  tensorflow/tensorflow/core/kernels/resize_bicubic_op.cc
    bicubic_coeffs = (
            (1, 0,     -(alpha+3), (alpha+2)),
            (0, alpha, -2*alpha,   alpha    ),
            (0, -alpha, 2*alpha+3, -alpha-2 ),
            (0, 0,      alpha,     -alpha   )
            )

    with tf.variable_scope('bilinear_interp'):
        batch_size, height, width, channels = im.get_shape().as_list()

        x = tf.cast(x, tf.float32)
        y = tf.cast(y, tf.float32)
        height_f = tf.cast(height, tf.float32)
        width_f = tf.cast(width, tf.float32)
        out_height = out_size[0]
        out_width = out_size[1]

        # scale indices from [-1, 1] to [0, width/height - 1]
        x = tf.clip_by_value(x, -1, 1)
        y = tf.clip_by_value(y, -1, 1)
        x = (x + 1.0) / 2.0 * (width_f-1.0)
        y = (y + 1.0) / 2.0 * (height_f-1.0)

        # do sampling
        # integer coordinates of 4x4 neighbourhood around (x0_f, y0_f)
        x0_f = tf.floor(x)
        y0_f = tf.floor(y)
        xm1_f = x0_f - 1
        ym1_f = y0_f - 1
        xp1_f = x0_f + 1
        yp1_f = y0_f + 1
        xp2_f = x0_f + 2
        yp2_f = y0_f + 2

        # clipped integer coordinates
        xs = [0]*4
        ys = [0]*4
        xs[0] = tf.cast(x0_f, tf.int32)
        ys[0] = tf.cast(y0_f, tf.int32)
        xs[1] = tf.cast(tf.maximum(xm1_f, 0), tf.int32)
        ys[1] = tf.cast(tf.maximum(ym1_f, 0), tf.int32)
        xs[2] = tf.cast(tf.minimum(xp1_f, width_f - 1),  tf.int32)
        ys[2] = tf.cast(tf.minimum(yp1_f, height_f - 1), tf.int32)
        xs[3] = tf.cast(tf.minimum(xp2_f, width_f - 1),  tf.int32)
        ys[3] = tf.cast(tf.minimum(yp2_f, height_f - 1), tf.int32)

        # indices of neighbours for the batch
        dim2 = width
        dim1 = width*height
        base = _repeat(tf.range(batch_size)*dim1, out_height*out_width)

        idx = []
        for i in range(4):
            idx.append([])
            for j in range(4):
                cur_idx = base + ys[i]*dim2 + xs[j]
                idx[i].append(cur_idx)

        # use indices to lookup pixels in the flat image and restore
        # channels dim
        im_flat = tf.reshape(im, tf.pack([-1, channels]))

        Is = []
        for i in range(4):
            Is.append([])
            for j in range(4):
                Is[i].append(tf.gather(im_flat, idx[i][j]))

        def get_weights(x, x0_f):
            tx = (x-x0_f)
            tx2 = tx * tx
            tx3 = tx2 * tx
            t = [1, tx, tx2, tx3]
            weights = []
            for i in range(4):
                result = 0
                for j in range(4):
                    result = result + bicubic_coeffs[i][j]*t[j]
                result = tf.reshape(result, [-1, 1])
                weights.append(result)
            return weights


        # to calculate interpolated values first, 
        # interpolate in x dim 4 times for y=[0, -1, 1, 2]
        weights = get_weights(x, x0_f)
        x_interp = []
        for i in range(4):
            result = []
            for j in range(4):
                result = result + [weights[j]*Is[i][j]]
            x_interp.append(tf.add_n(result))

        # finally, interpolate in y dim using interpolations in x dim
        weights = get_weights(y, y0_f)
        y_interp = []
        for i in range(4):
            y_interp = y_interp + [weights[i]*x_interp[i]]

        output = tf.add_n(y_interp)
        return output

