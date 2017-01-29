#!/usr/bin/env python
# coding:utf-8


import socio
import numpy as np
import sys
import random
import matplotlib.pyplot as plt
import os
from mpl_toolkits.axes_grid1 import make_axes_locatable
import argparse
import re
import st
from scipy.misc import imsave


def get_progressbar_str(progress, message=""):
    MAX_LEN = 50
    BAR_LEN = int(MAX_LEN * progress)
    return (message[0:5] + '[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) +
            '] %.1f%%' % (progress * 100.))


def div1(x, C, dim, smooth=5):
    sh = C.shape
    C = C.reshape(-1, 128)
    A = [st.golay(x, i, dim) for i in C]
    print np.max(A)
    return np.array(A).reshape(sh)
    # for i in range(C.shape[0]):
    #     progress = i/float(C.shape[0]-1)
    #     sys.stderr.write('\r\033[K' + get_progressbar_str(progress,message="golay"))
    #     sys.stderr.flush()
    #     for j in range(C.shape[1]):
    #         C[i][j]=st.golay(x,C[i][j],dim,m=smooth)

    # print
    # return C


def trns(A, B):
    with np.errstate(all='ignore'):
        return np.abs(A / (B))
    # C=np.copy(A)
    # for i in range(A.shape[0]):
    #     for j in range(A.shape[1]):
    #         try:
    #             C=A[i][j]/B[i][j]
    #         except:
    #             print A[i][j],B[i][j]
    # return C


def abs(A, light):
    with np.errstate(all='ignore'):
        C = np.log10(np.abs(light / A))
    return C


def check(ori, message="enter value [ s: skip , ctl+c : exit]> "):
    while True:
        print message,
        val = sys.stdin.readline()
        if(val[0] is "e" and val[1] is "x" and val[2] is "i" and val):
            exit()
        if(val[0] is "s"):
            return ori
        try:
            float(val)
            break
        except:
            print "#ERROR : enter value"
    return float(val)


def parse2slice(snapshot):

    def parsesection(snap):
        try:
            section = int(snap)
        except ValueError:
            section = [int(s) if s else None for s in snap.split(':')]
            if len(section) > 3:
                raise ValueError('snapshots input incorrect')
            section = slice(*section)
        return section

    section = [s if s else None for s in snapshot.split(',')]
    return tuple([parsesection(section[i]) for i in range(len(section))])


# command line arguments
parser = argparse.ArgumentParser(description='Convert float file to image')
parser.add_argument(dest='fname_float', metavar='FLOAT')
parser.add_argument('-o', '--out', type=str, default="img.png",
                    help='Specify output (ex. spec.png spec.txt DIR/out.png)')
parser.add_argument('-s', '--skip', type=int, default='1',
                    help='Reduce output by skipping')
parser.add_argument('-r', '--raw', action='store_true',
                    default=False, help='Write raw image')
parser.add_argument('-b', '--band', nargs='+', default=None,
                    help='Bands to write (ex. -b 800 900)')
parser.add_argument('-l', metavar='FLOAT')
parser.add_argument('-w', dest='wavelength')
parser.add_argument('-a', '--absorbance', action='store_true',
                    required=False, default=False)
parser.add_argument('-t', '--transmittance',
                    action='store_true', required=False, default=False)
parser.add_argument('-i', '--interactivemode', action='store_true',
                    required=False, default=False, help='interactivemode')

parser.add_argument('-c', '--PseudoColor',
                    action='store_true', required=False, default=False)
parser.add_argument('-m', '--movie', type=str, default=None)
parser.add_argument('-yx', '--ymax', metavar='YMAX',
                    type=float, required=False, default=None)
parser.add_argument('-ym', '--ymin', metavar='YMIN',
                    type=float, required=False, default=None)
parser.add_argument('-d1', '--div1', metavar='FIRST_DERIVATIVE',
                    type=int, required=False, nargs='?', default=0)
parser.add_argument('-d2', '--div2', metavar='SECOND_DERIVATIVE',
                    type=int, required=False, nargs='?', default=0)
parser.add_argument('-sl', '--slice', type=str,
                    required=False, default=":,:,:")
args = parser.parse_args()


if args.fname_float.split(".")[-1] == "float":

    hsi = socio.openfloat(args.fname_float)
    x = hsi.getWavelengths()

    if args.l:
        light = socio.openfloat(args.l)
        B = light[parse2slice(args.slice)]
        # B=light[5:10,20:40,:]

    x = x[parse2slice(args.slice)[2]]
    A = hsi[parse2slice(args.slice)]
    C = A

    if args.transmittance and args.absorbance:
        "#ERROR :"

    if args.transmittance:
        C = trns(A, B)

elif args.fname_float.split(".")[-1] == "pkl":
    C = st.loadpickle(args.fname_float)
    try:
        x = np.loadtxt(args.wavelength)
    except:
        x = range(C.shape[2])#[parse2slice(args.slice)[2]]

    C=C[parse2slice(args.slice)]
    x=x[parse2slice(args.slice)[2]]

if args.absorbance:
    if not args.l:
        print "#ERROR : Specify light file."
        exit()
    C = abs(A, B)

if args.div1 != 0:
    m = args.div1
    if(args.div1 is None):
        m = 5
    C = div1(x, C, 1, smooth=m)

if args.div2 != 0:
    m = args.div2
    if(args.div2 is None):
        m = 5
    C = div1(x, C, 2, smooth=m)


outname = args.out
minn = C.min()
maxx = C.max()
if args.transmittance:
    minn = 0.
    maxx = 1.
if args.absorbance:
    minn = 0.
    maxx = 2.

while True:

    print "set (min,maxx) = (%lf,%lf)" % (minn, maxx)
    plt.figure()

    if args.interactivemode:
        # plt.figure()
        xlis = [random.randint(0, A.shape[0] - 1) for i in range(30)]
        ylis = [random.randint(0, A.shape[1] - 1) for i in range(30)]
        for i, j in zip(xlis, ylis):
            plt.plot(x, C[i][j])
        plt.show()

    if args.ymin is not None:
        minn = args.ymin
    if args.ymax is not None:
        maxx = args.ymax
    if args.interactivemode:
        minn = check(
            minn, message="enter MIN value [ s: skip , ctl+c : exit] > ")
        maxx = check(
            maxx, message="enter MAX value [ s: skip , ctl+c : exit] > ")

    # if len(C.shape)!=3:
    #     C=C[:,:,np.newaxis]
    #     FLAG=True
    # print x
    # exit()
    try:
        A = int(x)
        A = x
        x = []
        x.append(A)
        C = C[:, :, np.newaxis]
    except:
        pass

    for i, band in enumerate(x):
        # save
        # make file name
        # print band
        if outname is not None:
            fname_out_base = outname.split('.')[0]
            DIR = re.sub(r'[^/]+$', '', fname_out_base)
            # make non-existing dir
            if DIR is not '' and not os.path.exists(fname_out_base.split('/')[-2]):
                os.mkdir(fname_out_base.split('/')[-2])
            fname_out = fname_out_base + '_' + str(int(band)).zfill(4)
            suffix = '.' + outname.split('.')[-1]
            fname_out = fname_out + suffix

        else:
            # fname_out = fname_base + '_' + str(int(band)).zfill(4)
            # suffix = '.png'
            print "#ERROR: Specify suffix. [-o]"

        # save image
        if suffix == ".png":
            if (args.PseudoColor):
                img = C[:, :, i]
                print "write to " + fname_out, minn, maxx
                socio.savefig(fname_out, img, max=maxx, min=minn, color="c")
            else:
                img = C[:, :, i]
                img -= minn
                img = img / maxx
                img *= 65535.
                print "write to " + fname_out, minn, maxx
                imsave(fname_out, img.astype(np.uint16))
                # socio.savefig(fname_out, img,max=maxx,min=minn)

        if suffix == '.txt':
            if args.PseudoColor:
                print "##Warning : -c option is invalid"
            suffix = ".txt"
            img = C[:, :, i]
            print "write to " + fname_out
            np.savetxt(fname_out, img, fmt="%5.4lf")

    # os.system("open %s"%fname_out_base.split('/')[-2])
    if (args.movie) is not None:
        print args.movie
        print fname_out_base + "*" + suffix
        os.system("SocMakeMovie.py %s -f 10 -o %s" %
                  (fname_out_base + "*" + suffix, args.movie))
    if args.ymax is not None and args.ymin is not None and not args.interactivemode:
        exit()
    if not args.interactivemode:
        exit()
