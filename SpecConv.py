#!/usr/bin/env python


import numpy as np
import argparse
import re
from scipy.interpolate import splev, splrep
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import st


def isd(N):
    return bool(re.compile("^[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?$|nan$|^[-+]?inf$").match(N))


def parseSpectrum(filename):
    file_spec = open(filename)

    spectrum = []
    for line_raw in file_spec.readlines():
        line = line_raw.strip()

        spec = line.split()
        if isd(spec[0]) and isd(spec[1]):
            spectrum.append([float(spec[0]), float(spec[1])])
    return np.array(spectrum)


def thinouted(lst, newsize):
    result = []
    cnt = 0
    for x in lst:
        cnt -= newsize
        if cnt < 0:
            cnt += len(lst)
            result.append(x)
    return np.array(result)


# Parse program arguments
parser = argparse.ArgumentParser(description='Convert spectrum')
parser.add_argument(dest='file_in', nargs='+', metavar='SOURCE')
parser.add_argument('-o', '--out-suffix', metavar='SUFFIX',
                    type=str, default='conv')
parser.add_argument('-m', '--multiply', metavar='MULTIPLY',
                    type=float, required=False, default=1.0)
parser.add_argument('-mv', '--avr-num', metavar='AVERAGE',
                    type=int, required=False, default=1)
parser.add_argument('-sm', '--smooth', metavar='KERNEL_SIZE',
                    type=float, required=False, default=1.0)
parser.add_argument('-ks', '--kernel_size', metavar='KERNEL_SIZE_GAUSS',
                    type=int, required=False, nargs='?', default=10)
parser.add_argument('-pca', '--PCA', action='store_true',
                    required=False, default=False)


parser.add_argument('-l', '--log', action='store_true',
                    required=False, default=False)
parser.add_argument('-t', '--transpose', action='store_true',
                    required=False, default=False)
parser.add_argument('-d0', '--div0', metavar='ZERO_DERIVATIVE',
                    type=int, required=False, nargs='?', default=0)
parser.add_argument('-d1', '--div1', metavar='FIRST_DERIVATIVE',
                    type=int, required=False, nargs='?', default=0)
parser.add_argument('-d2', '--div2', metavar='SECOND_DERIVATIVE',
                    type=int, required=False, nargs='?', default=0)


parser.add_argument('-i', '--invert', action='store_true',
                    required=False, default=False)

parser.add_argument('-d', '--dark', metavar='DARK_SPECTRUM',
                    required=False, default=None)
args = parser.parse_args()


# parse files
spec_in = np.array(map(parseSpectrum, args.file_in))


if args.PCA:
    pca = PCA()
    pca.fit(np.array(spec_in[:, :, 1]))
    M = pca.components_
    E = pca.explained_variance_ratio_
    Esum = np.cumsum(E)[::-1][0]

    for i, spec in enumerate(spec_in):
        spec_out = np.copy(spec)
        spec_out[:, 1] = M[i]

        # save
        file_out = args.file_in[i].split('/')[-1]
        file_out = re.sub(r'_[a-zA-Z0-9]+\.[a-zA-Z0-9]+', '', file_out)
        file_out += '_' + str("%0.1f" % E[i]) + '_' + "pca" + '.txt'
        np.savetxt(file_out, spec_out, fmt='%12.8f')
        print 'Saved to ' + file_out
    print
    exit()


# dark substruct
if args.dark is not None:
    spec_dark = parseSpectrum(args.dark)


for i, spec in enumerate(spec_in):
    # multiply
    spec_out = np.copy(spec)
    # print spec_out[:,1:2]
    spec_out[:, 1:2] = spec_out[:, 1:2] * args.multiply

# moving average
    if args.avr_num != 1:
        kernel = np.ones((1, args.avr_num))[0] / args.avr_num
        spec_out[:, 1] = np.convolve(spec_out[:, 1], kernel, 'same')

# smoothing
# use gaussian kernel
    if args.smooth != 1.0 or args.kernel_size != 10:
        kernel_size = args.kernel_size  # len(spec_out[:,1])/2
#    kernel_size = len(spec_out[:,1])/2
        kernel = range(-kernel_size, kernel_size)
        kernel = np.array(kernel)
        kernel = -0.5 * kernel * kernel / (args.smooth * args.smooth)
        kernel = np.exp(kernel)
        kernel /= np.sum(kernel)
        A = np.convolve(spec_out[:, 1], kernel, 'same')
        # print A.shape,spec_out.shape
        spec_out[:, 1] = A
# derivate
# use laplacian kernel
    # if args.derivate2nd:
    #     kernel = [1,-2,1]
    #     kernel = np.array(kernel)
    #     kernel = kernel / (spec_out[1,0]-spec_out[0,0])**2
    #     spec_out[:,1] = np.convolve(spec_out[:,1], kernel, 'same')

    # if args.divegence:
    #     aa,bb=div(spec[:,0],spec[:,1])
    #     spec_out=np.array([aa,bb]).T

    if args.div0 != 0:
        m = args.div0
        if(args.div0 is None):
            m = (0 + 1) / 2
        # print m
        x = np.array(spec).T[0]
        y = np.array(spec).T[1]
        YY = st.golay(x, y, 0, m=m)
        spec_out = np.array([x, YY]).T

    if args.div1 != 0:
        m = args.div1
        if(args.div1 is None):
            m = (1 + 1) / 2
        x = np.array(spec).T[0]
        y = np.array(spec).T[1]
        YY = st.golay(x, y, 1, m=m)
        spec_out = np.array([x, YY]).T

    if args.div2 != 0:
        m = args.div2
        if(args.div2 is None):
            m = (2 + 1) / 2
        x = np.array(spec).T[0]
        y = np.array(spec).T[1]
        YY = st.golay(x, y, 2, m=m)
        spec_out = np.array([x, YY]).T

    if args.invert:
        spec_out[:, 1] = 1.0 / spec[:, 1]

# logarithm
    if args.log:
        spec_out[:, 1:2][spec_out[:, 1:2] <= 0] = 0.000000001
        spec_out[:, 1:2] = np.log10(spec_out[:, 1:2])

# transpose
    if args.transpose:
        spec_out = np.transpose(spec_out)

# dark substruct
    if args.dark is not None:
        spec_out[:, 1] -= spec_dark[:, 1]

# save
    file_out = st.mkdir_suff(args.out_suffix, "conv",
                             base=args.file_in[i], ex=".txt")
    np.savetxt(file_out, spec_out, fmt='%12.8f')
    print 'Saved to ' + file_out

print
