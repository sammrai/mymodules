import re
import numpy as np
import pickle
import os

flatten=lambda i,d=-1:[a for b in i for a in(flatten(b,d-(d>0))if hasattr(b,'__iter__')and d else(b,))]

def loadpickle(filename):
    f = open(filename)
    h = pickle.load(f)
    f.close()
    return h


def savepickle(filename, instance):
    f = open(filename, 'w')
    h = pickle.dump(instance, f)
    f.close()


def mkdir(DIR, rm=False):

    if rm:
        import shutil
        try:
            shutil.rmtree(DIR)
        except:
            os.makedirs(DIR)
    try:
        os.makedirs(DIR)
    except:
        return 0


def rm(DIR):
    import shutil
    shutil.rmtree(DIR)


def mkdir_suff(f, suff, base="", ex=""):
    """
    out suffix maker.

    Examples
    ----------------------------

    If argument f is not set, suffix is set to suff.

    >>> st.mkdir_suff("","suff",base="base",ex=".txt")
    base_suff.txt

    If argument f is set only directry, suffix is set to base and make dir automatically.

    >>> st.mkdir_suff("dir/","suff",base="base",ex=".txt")
    dir/base_suff.txt

    If argument f is set, f is interpreted as suffix.

    >>> st.mkdir_suff("test","suff",base="base",ex=".txt")
    base_test.txt

    If argument f is set with dir, after dir/ argument is interpreted as suffix and make dir automatically.

    >>> st.mkdir_suff("dir/test","suff",base="base",ex=".txt")
    dir/base_test.txt

    If argument f is set with extantion, return f.

    >>> st.mkdir_suff("test.dat","suff",base="base",ex=".txt")
    test.dat


    """
    try:
        if len(f.split(".")) > 1:
            file_out = f
            DIR = os.path.dirname(f) + "/"
            if not os.path.exists(DIR):
                os.mkdir(DIR)
            return file_out
    except:
        pass

    if os.path.basename(f) is "":
        BASE = suff
    else:
        BASE = os.path.basename(f)
    if os.path.dirname(f) is "":
        DIR = ""
    else:
        DIR = os.path.dirname(f) + "/"
        if not os.path.exists(DIR):
            os.mkdir(DIR)
    file_out = re.sub(r'_Sub.*..*$', '', os.path.basename(base))
    file_out = DIR + re.sub(r'\..*$', '', file_out) + "_" + BASE + ex
    return file_out


def convert2new_axis(x, y, new_x):
    from scipy import signal, interpolate
    from scipy.interpolate import splev, splrep
    y_ = np.copy(y)
    x_ = np.copy(x)

    # index =np.where(((np.isinf(y_))+(np.isnan(y_)))==True)[0]
    # x_=np.delete(x_,index)
    # y_=np.delete(y_,index)

    mn, mx = np.searchsorted(new_x, x_[0]), np.searchsorted(new_x, x_[-1]) - 1
    ret = [0. for i in new_x]

    # tck=splrep(x_,y_)
    tck = interpolate.interp1d(x_, y_)
    # ret[mn:mx] = splev(new_x[mn:mx],tck)
    ret[mn:mx] = tck(new_x[mn:mx])
    return ret


def arrange(y):
    # y=np.copy(y_)
    index = np.where(((np.isinf(y)) + (np.isnan(y))) == True)[0]
    y[index] = 0
    return y


def golay(*args, **kwargs):
    # from scipy.interpolate import splev, splrep
    from scipy import signal, interpolate

    """

    This function provide dim-order differential smoothness filter.

    Examples
    ====

    >>> import st
    >>> y_ = st.golay(y,2) #second derivatives
    >>> y_ = st.golay(x,y,2) #second derivatives (with any x)
    >>> y_ = st.golay(x,y,0) #remove invalid values. (nan,inf)
    >>> y_ = st.golay(x,y,1,m=5) #apply smoothness


    Parametors
    -----------------

    x : array-like, shape (n_samples,) , optional
        x-axis data. The pitch size of x dont have to set evenly.
    y : array-like, shape (n_samples,)
        y-axis data
    dim : int
        dim-order differential.
    m : int
        smoothness. In technical, the polynomial order m must be less than the frame size x.
    mode : {valid', 'same'}, optional
        'same':
          By default, mode is 'same'. Mode 'same' returns output of length ``n_samples``.  Boundary
          effects are still visible.
        'valid':
          Mode 'valid' returns output of length ``n_samples-2*m``. Values outside the signal boundary will be cut off.

    Returns
    -----------------

    y_ : array-like, shape (n_samples,)
        Golay returns the array of dim-order differencial with m somoothness

    Reference
    ====
    Golay filter is one of the method of Savitzky-Golay.In ordinaly, See also http://www.mathworks.com/help/signal/ref/sgolay.html?s_tid=gn_loc_drop
    http://www.empitsu.com/pdf/sgd.20081106.pdf
    """

    # Read argumants
    if len(args) == 3:
        x = np.copy(args[0])
        y = np.copy(args[1])
        dim = args[2]
    elif len(args) == 2:
        y = np.copy(args[0])
        dim = args[1]
        x = np.copy(range(len(y)))
    else:
        raise Exception(
            "golay() takes at least 2 arguments (%d given)" % len(args))

    if "m" in kwargs:
        m = kwargs["m"]
    else:
        m = (dim + 1) / 2
    if "mode" in kwargs:
        mode = kwargs["mode"]
    else:
        mode = "same"

    # Check data
    if len(x) != len(y):
        raise ValueError("Input sequence length is invalid %d %d" %
                         (len(x), len(y)))
    if len(x) - m * 2 <= 3:
        raise ValueError("Input sequence length is too small. %d" % len(x_))
    if m < (dim + 1) / 2:
        raise ValueError("Smoothness value :m takes at most %d" %
                         ((dim + 1) / 2))
    if m == 0 and dim == 0:
        return y
    # remove invalid value.
    y = arrange(y)

    # Leveling new_x interavl.
    tck = interpolate.interp1d(x, y)

    alpha = 1.
    x_temp = np.linspace(min(x), max(x), len(x) * alpha)
    while not(x_temp[m:len(x_temp) - m][0] < x[m] and x[len(x) - m - 1] < x_temp[m:len(x_temp) - m][-1]):
        x_temp = np.linspace(min(x), max(x), len(x) * alpha)
        alpha *= 1.5
    y = tck(x_temp)

    def diff(_x, _y, _dim, _m):
        dx = _x[1] - _x[0]
        array = []
        k = _dim  # the number of times of differential
        for ik in range(k + 1):
            a = []
            for im in range(-_m, _m + 1):
                a.append(pow(im, ik))
            array.append(a)
        X = np.array(array)
        B = np.dot(np.linalg.inv(np.dot(X, X.T)), X)

        func = lambda s: (lambda m: m(m)(s))(lambda proc: (
            lambda n: 1 if n == 0 else n * proc(proc)(n - 1)))
        seeknum = len(_y) - 2 * _m
        AA = [np.dot(B, _y[seek:seek + 1 + _m * 2]) *
              func(dim) / dx**_dim for seek in range(seeknum)]
        xx = [_x[seek + _m] for seek in range(seeknum)]
        return np.array(xx), np.array(AA).T[dim]

    new_x, new_y = diff(x_temp, y, dim, m)
    # tck=interpolate.InterpolatedUnivariateSpline(new_x,new_y,)
    tck = interpolate.interp1d(new_x, new_y,)
    # print (new_x[0]<x[m] , x[len(x)-m-1]<new_x[-1])
    y = tck(x[m:len(x) - m])

    if mode is "same":
        ret = np.array([0. for i in range(len(x))])
        ret[m:len(x) - m] = y
        return ret
    if mode is "valid":
        return np.array(y)

    # return thinouted(np.array(xx),tnum),thinouted(np.array(AA).T[dim],tnum)
    # return np.array(xx),np.array(AA).T[dim]
