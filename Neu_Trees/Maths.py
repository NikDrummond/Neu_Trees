# tools for thresholding image data
import numpy as np

import warnings
warnings.filterwarnings("ignore",category=RuntimeWarning)


# cumulative sum
csum = lambda z:np.cumsum(z)[:-1]
#de-cumulative sum
dsum = lambda z:np.cumsum(z[::-1])[-2::-1]

# Use the mean for ties .
argmax = lambda x , f:np.mean(x[:-1][f==np.max(f)]) 
clip = lambda z:np.maximum(1e-30,z) 

def check_symmetric(a, approx = False, **kwargs):
    """
    Check if a matrix is symetrical across it's diagonal. If approx is false np.allclose is used to determine approximate similarity within specified bounds.
    
    Parameters
    ----------

    a:          np.array
        an n*n np.array

    approx:     bool
        if True, checks for exact symetry

    **kwargs    kwargs
        kwargs passed to np.allclose if approx is True. See np.allclose for possible arguments.


    Returns
    -------

    bool
        False if a is not symetric, True otherwise

    """
    
    if approx == True:
        return np.allclose(a, a.T, **kwargs)
    else:
        return np.all(a == a.T)

def double_MADs(x,Threshold = 3.5,cc = 0.6):
    """
    Performs double median absolute difference Thresholding on a 1-dimensional vector.

    Parameters
    ----------

    x:          list | np.array
        1d vector to threshold

    Threshold:  float
        Threshold value in standard deviations

    cc:    float
        constant applied to the modified z-scores. If 1d vector is ~ normally distributed, this should be ~1.5. If the data is heavily left skewed, values lower than 1 are better.

    Returns
    -------

    np.array:
        array of values in x above threshold
    """

    # make sure x is an array
    if not isinstance(x,np.ndarray):
        x = np.array(x)

    # get the median
    m = np.median(x)
    # the absolute difference
    abs_dev = np.abs(x-m)
    #?
    left_mad = np.median(abs_dev[x <= m])
    right_mad = np.median(abs_dev[x >= m])

    x_mad = left_mad * np.ones(len(x))
    x_mad[x>m] = right_mad

    modified_z_score = cc * abs_dev / x_mad

    modified_z_score[x == m] = 0

    return x[modified_z_score > Threshold]

def hist_array(N, range = None):
    """
    Given an input array, generate counts and bin centers. For a given array with a maximum value of x, and minimum value of y, the returned histogram will have at least (x - y) + 1 bins. If range is specified, the histogram bin values will be arranged between this minimum and maximum values. If range is not passed, bin values will be between 0 and x + 1.

    Parameters
    ----------

    N:      np.ndarray
        input array - will be flattened and converted to integers


    range:  tuple | list
        Either a tuple or a list of two values to be used as the maximum and minimum values of the input histogram.


    Returns
    -------

    n:      np.ndarray
        Frequency values - number of occurrences within each bin


    x:      np.ndarray
        Array of starting values of each bin

    """
    image = N.ravel().astype(int)
    n = np.bincount(image,minlength = np.max(image) - np.min(image) + 1)
    if range != None:
        if len(range) != 2:
            raise TypeError('Range is not length 2')
        else:
            x = np.arange(range[0], range[1])    
    elif range == None:
        x = np.arange(0,np.max(image) + 1)
    else:
        raise TypeError("range input type not supported")
    
    return n,x

def preliminaries(n, x):
    """ Some math that is shared across each algorithm ."""
    assert np . all(n >= 0)
    x = np.arange(len(n), dtype=n.dtype) if x is None else x
    assert np . all(x[1:] >= x[: -1])
    w0 = clip(csum(n))
    w1 = clip(dsum(n))
    p0 = w0/(w0 + w1)
    p1 = w1/(w0 + w1)
    mu0 = csum(n*x)/w0
    mu1 = dsum(n*x)/w1
    d0 = csum(n*x**2)-w0*mu0**2
    d1 = dsum(n*x**2)-w1*mu1**2
    return x, w0, w1, p0, p1, mu0, mu1, d0, d1

def Otsu(n, x):
    """ Otsu method ."""
    x, w0, w1, _, _, mu0, mu1, _, _ = preliminaries(n, x)
    o = w0*w1*(mu0-mu1)**2
    return argmax(x,o),o

def MET(n, x):
    """ Minimum Error Thresholding."""
    x, w0, w1, _, _, _, _, d0, d1 = preliminaries(n, x)
    ell = (1+w0*np.log(clip(d0/w0))+w1*np.log(clip(d1/w1))
        - 2 * (w0*np.log(clip(w0))+w1*np.log(clip(w1))))
    return argmax(x,-ell), ell  # argmin ()

def wprctile(n, x, omega=0.5):
    """ Weighted percentile, with weighted median as default ."""
    assert omega >= 0 and omega <= 1
    x, _, _, p0, p1, _, _, _, _ = preliminaries(n, x)
    h = -omega*np.log(clip(p0))-(1.-omega)*np.log(clip(p1))
    return argmax(x,-h),h  # argmin ()

def GHT(n, x, nu=None, tau=0.1, kappa=0.1, omega=0.5):
    """ Our generalization of the above algorithms ."""
    if nu == None:
        nu = abs(len(x)/2)
    assert nu >= 0
    assert tau >= 0
    assert kappa >= 0
    assert omega >= 0 and omega <= 1
    x, w0, w1, p0, p1, _, _, d0, d1 = preliminaries(n, x)
    v0 = clip((p0*nu*tau**2+d0)/(p0*nu+w0))
    v1 = clip((p1*nu*tau**2+d1)/(p1*nu+w1))
    f0 = - d0/v0-w0*np.log(v0) + 2*(w0+kappa*omega)*np.log(w0)
    f1 = - d1/v1-w1*np.log(v1)+2*\
        (w1+kappa*(1-omega))*np.log(w1)
    return argmax(x, f0+f1), f0+f1

def edf(data, alpha=.05, x0=None, x1=None , bins = 100):
    """
    Calculate the empirical distribution function and confidence intervals around it.

    Parameters
    ----------
    data:

    alpha:

    x0:

    x1:

    Returns
    -------
    x:

    y:

    l:

    u:
    """


    x0 = data.min() if x0 is None else x0
    x1 = data.max() if x1 is None else x1
    x = np.linspace(x0, x1, bins)
    N = data.size
    y = np.zeros_like(x)
    l = np.zeros_like(x)
    u = np.zeros_like(x)
    e = np.sqrt(1.0/(2*N) * np.log(2./alpha))
    for i, xx in enumerate(x):
        y[i] = np.sum(data <= xx)/N
        l[i] = np.maximum( y[i] - e, 0 )
        u[i] = np.minimum( y[i] + e, 1 )
    return x, y, l, u

def rotation_matrix_3D(theta1, theta2, theta3, order='xyz'):
    """
    input
        theta1, theta2, theta3 = rotation angles in rotation order (degrees)
        order = rotation order of x,y,z: e.g. XZY rotation -- 'xzy'
    output
        3x3 rotation matrix (numpy array)
    """
    c1 = np.cos(theta1 * np.pi / 180)
    s1 = np.sin(theta1 * np.pi / 180)
    c2 = np.cos(theta2 * np.pi / 180)
    s2 = np.sin(theta2 * np.pi / 180)
    c3 = np.cos(theta3 * np.pi / 180)
    s3 = np.sin(theta3 * np.pi / 180)

    if order == 'xzx':
        matrix=np.array([[c2, -c3*s2, s2*s3],
                         [c1*s2, c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3],
                         [s1*s2, c1*s3+c2*c3*s1, c1*c3-c2*s1*s3]])
    elif order=='xyx':
        matrix=np.array([[c2, s2*s3, c3*s2],
                         [s1*s2, c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1],
                         [-c1*s2, c3*s1+c1*c2*s3, c1*c2*c3-s1*s3]])
    elif order=='yxy':
        matrix=np.array([[c1*c3-c2*s1*s3, s1*s2, c1*s3+c2*c3*s1],
                         [s2*s3, c2, -c3*s2],
                         [-c3*s1-c1*c2*s3, c1*s2, c1*c2*c3-s1*s3]])
    elif order=='yzy':
        matrix=np.array([[c1*c2*c3-s1*s3, -c1*s2, c3*s1+c1*c2*s3],
                         [c3*s2, c2, s2*s3],
                         [-c1*s3-c2*c3*s1, s1*s2, c1*c3-c2*s1*s3]])
    elif order=='zyz':
        matrix=np.array([[c1*c2*c3-s1*s3, -c3*s1-c1*c2*s3, c1*s2],
                         [c1*s3+c2*c3*s1, c1*c3-c2*s1*s3, s1*s2],
                         [-c3*s2, s2*s3, c2]])
    elif order=='zxz':
        matrix=np.array([[c1*c3-c2*s1*s3, -c1*s3-c2*c3*s1, s1*s2],
                         [c3*s1+c1*c2*s3, c1*c2*c3-s1*s3, -c1*s2],
                         [s2*s3, c3*s2, c2]])
    elif order=='xyz':
        matrix=np.array([[c2*c3, -c2*s3, s2],
                         [c1*s3+c3*s1*s2, c1*c3-s1*s2*s3, -c2*s1],
                         [s1*s3-c1*c3*s2, c3*s1+c1*s2*s3, c1*c2]])
    elif order=='xzy':
        matrix=np.array([[c2*c3, -s2, c2*s3],
                         [s1*s3+c1*c3*s2, c1*c2, c1*s2*s3-c3*s1],
                         [c3*s1*s2-c1*s3, c2*s1, c1*c3+s1*s2*s3]])
    elif order=='yxz':
        matrix=np.array([[c1*c3+s1*s2*s3, c3*s1*s2-c1*s3, c2*s1],
                         [c2*s3, c2*c3, -s2],
                         [c1*s2*s3-c3*s1, c1*c3*s2+s1*s3, c1*c2]])
    elif order=='yzx':
        matrix=np.array([[c1*c2, s1*s3-c1*c3*s2, c3*s1+c1*s2*s3],
                         [s2, c2*c3, -c2*s3],
                         [-c2*s1, c1*s3+c3*s1*s2, c1*c3-s1*s2*s3]])
    elif order=='zyx':
        matrix=np.array([[c1*c2, c1*s2*s3-c3*s1, s1*s3+c1*c3*s2],
                         [c2*s1, c1*c3+s1*s2*s3, c3*s1*s2-c1*s3],
                         [-s2, c2*s3, c2*c3]])
    elif order=='zxy':
        matrix=np.array([[c1*c3-s1*s2*s3, -c2*s1, c1*s3+c3*s1*s2],
                         [c3*s1+c1*s2*s3, c1*c2, s1*s3-c1*c3*s2],
                         [-c2*s3, s2, c2*c3]])

    return matrix

def coords_Eig(Coords, center = False, PCA = False,):
    """

    Performs Eigen Decomposition on a given array of coordinates. This is done by calculating the covariance matrix of the coordinates array, then the eigenvectors and values of this coordinate matrix.

    Parameters
    ----------
    
    Coords:         np.array
        Coordinate array from which to calculate eigenvectors and values. This should be in the form n x d, where each row is an observation, and column a dimension.

    center:         Bool
        Whether or not to mean center the data before computing

    PCA:            Bool
        Whether or not to return eigenvalues as a fraction of the sum of all eigenvalues, as in PCA, showing variance explained.

    Returns
    -------

    evals:          np.array
        eigenvalues, ordered from largest to smallest

    evects:         list
        List of np.arrays, each of witch is the eigenvector corresponding to the descending order of eigenvalues
    
    """

    # check dimensions of input

    # mean center the data, if we want
    if center == True:
        for i in range(Coords.shape[0]):
            Coords[i] -= np.mean(Coords[i])

    cov_mat = np.cov(Coords)
    evals, evects = np.linalg.eig(cov_mat)
    # sort largest to smallest
    sort_inds  = np.argsort(evals)[::-1]

    if PCA == True:
        evals /= sum(evals)

    evects = [evects[:,i] for i in sort_inds]

    return evals[sort_inds], evects


def bootstrap(data, sample_k = 100000, size = None,statistic = 'Mean', p_calc = '0.95', ci = '0.95',returns = 'distribution'):
    """
    Performs bootstrapping on given data

    Parameters
    ----------

    data:           np.array   
            The data from which to draw the bootstrap sample from. 

    sample_k:      int
            The number of bootstraps/samples to be drawn.

    size:          None | int
            Size of each bootstrap sample to be drawn from the original data with replacement. if None (default) the size of the original sample is used. Alternatively, an integer value can be passed which is <= the size of the original sample. if the given integer is larger than the original sample size, an error is returned.

    statistic:     str
            The sample statistic to be bootstrapped. Default is 'Mean'

            Available options:
                'Mean'  : The mean of each sample is calculated for the bootstrapped distribution
                'Median': The median of each sample is calculated for the bootstrapped distribution
                'Std'   : The standard deviation of each sample is calculated for the bootstrapped distribution
                'Var'   : The variance of each sample is calculated for the bootstrapped distribution

    ci:            float
            The confidence level of the confidence interval. default is 0.95

    alpha:         float
            alpha level for a returned p-value of the observed data against the bootstrapped sample.

    returns:      str
            Determines what is returned by the function. default is 'Full'. This returns the bootstrapped distribution, as well as a confidence interval, p value, and standard error of the bootstrapped distribution. Alternatively, only the confidence interval, p value, and standard error can be returned using 'Statistics'; or 'Distribution' returns only the bootstrapped distribution


    Returns
    -------

        distribution:   list
                The bootstrapped distribution of the measured statistic

        ci:             tuple
                The confidence interval of the bootstrapped distribution, using the percentile method. Default is 0.95. A two-tailed confidence interval is calculated, so if the with the example of the default setting (0.95) the top and bottom 0.25 are used to calculate the confidence interval.

        p:              float
                The p-value of the observed distribution against the bootstrapped distribution

        se:             float
                the bootstrap standard error - the sample standard deviation of the bootstrap distribution

    """

    # sort out which 'statistic' we are bootstrapping

    if isinstance(size, int):
        if size > len(data):
            raise ValueError("Given size is larger than the original data set")
    else:
        size = len(data)

    if statistic == 'Mean':
        test = np.mean
    elif statistic == 'Median':
        test = np.median
    elif statistic == 'Std':
        test = np.std
    elif statistic == 'Var':
        test = np.var    

    # perform bootstrapping
    dist = np.zeros(sample_k)
    for i in range(sample_k):
        x = np.random.choice(data,size = size, replace = True)
        dist[i] = test(x)

    # Work out bits which will be returned

    if returns == 'distribution':
        return dist
    else:
        pass

def eig_axis_eulers(evects):
    """
    Given a list of eigenvector as returned by coords_Eig, return Euler angles needed to align The first eigenvector with the y-axis, second with the x-axis, and third with the z-axis
    """
    # Yaw
    theta1 = np.rad2deg(np.arctan(evects[0][0]/evects[0][1]))
    # pitch
    theta2 = np.rad2deg(np.arctan(evects[1][2]/evects[1][0]))
    # roll
    theta3 = np.rad2deg(np.arctan(evects[2][1]/evects[2][2]))

    return theta1, theta2, theta3

def snap_to_axis(coords, error_tol = 0.0000001, return_theta = False):
    """
    Given a set of 3D coordinates, rotates the coordinates so the Eigenvectors align with the original coordinate system axis. This is done so the first Eigenvector (corresponding to the highest eigenvalue) aligns with the y-axis, the second to the x-axis, and the third to the z-axis. Rotation is done in 'zyx' order, Rotating first around the z-axis to align the first eigenvector to the y-axis (this is Yaw), Second around the y-axis to align the second eigenvector to the x-axis (Pitch), and finally around the x-axis to align the final eigenvector to the z-axis (roll). 
    
    Parameters
    ----------

    coords:         np.array
        dimensions by observations np.array with coordinates. Function can only accept 3D coordinates currently

    error_tol:      float
        Some error around how closely the eigenvectors can align to the image axis seems to be introduced (at this stage, it is unclear why this is the case...). The error_tol parameter sets a threshold where by the rotation will be interatively re-rotated, new eigenvectors calculated, and euler angles calculated, until the euler angles are less than this threshold.

    return_theta:   Bool
        If True, final euler angles are returned, which will be less than error_tol. 

    Returns
    -------

    r_coords:       np.array
        Rotated coordinate array, the same shape as the input coordinate array

    thetas:         list
        list of euler angles of the final rotation, which will be less than error_tol. Angles are ordered in the order of rotations.
    
    
    
    """
    ### Rotation - Yaw, Pitch, and Roll
    evals,evects = coords_Eig(coords)

    theta1, theta2, theta3 = eig_axis_eulers(evects)

    R = rotation_matrix_3D(theta1,theta2,theta3, order = 'zyx')
    r_coords = R @ coords

    # Check and correct for error
    # get "final" angles
    evals,evects = coords_Eig(r_coords)

    theta1, theta2, theta3 = eig_axis_eulers(evects)

    while (abs(theta1) > error_tol) and (abs(theta2) > error_tol) and (abs(theta3) > error_tol):

        evals,evects = coords_Eig(r_coords)
        # pitch
        theta1, theta2, theta3 = eig_axis_eulers(evects)

        R = rotation_matrix_3D(theta1,theta2,theta3, order = 'zyx')
        r_coords = R @ r_coords

    if return_theta == False:
        return r_coords
    else:
        return r_coords, [theta1,theta2,theta3]

def scale(x, rmin, rmax, tmin, tmax):
    """
    linearly scale x to the interval between tmin and tmax

    Parameters
    ----------
    x:      float | int | list | np.array
        single value, or set of values you wish to scale to given interval
    
    rmin:   int | float
        minimum value in observed range

    rmax:   int | float
        max value in observed range

    tmin:   int | float
        minimum of range for x to be scaled to

    tmax:   int | float
        maximum of range for x to be scaled to


    Returns
    -------
    scaled_x:   float | int | list | np.array
        x, linearly scaled between tmin and tmax. returned in the same object type as x

    """
    if min == None:
        rmin = np.min(x)
    else:
        rmin = rmin
    if max == None:
        rmax = np.max(x)
    else:
        rmax = rmax

    convert = False
    if isinstance(x,list):
        convert = True
        x = np.array(x)

    scaled = ((x - rmin)/(rmax-rmin)) * (tmax - tmin) + tmin

    if convert == True:
        scaled = list(scaled)

    return scaled
