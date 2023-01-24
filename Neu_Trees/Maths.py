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

def edf(data, alpha=.05, x0=None, x1=None ):
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
    x = np.linspace(x0, x1, 100)
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



    
