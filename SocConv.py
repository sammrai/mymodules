#!/usr/bin/env python

import os
import argparse
import numpy as np
import socio
import st


def div1(x, C, dim, smooth=5):
    sh = C.shape
    C = C.reshape(-1, 128)
    A = [st.golay(x, i, dim) for i in C]
    print np.max(A)
    return np.array(A).reshape(sh)


def trns(A, light):
    with np.errstate(all='ignore'):
        return np.abs(A / (light))


def abs(A, light):
    with np.errstate(all='ignore'):
        C = np.log10(np.abs(light / A))
    return C


def get_hsi_wav(filename, wav=None):
    if filename.split(".")[-1] == "float":

        hsi = socio.openfloat(filename)
        x = hsi.getWavelengths()

    elif filename.split(".")[-1] == "pkl":
        hsi = st.loadpickle(filename)
        try:
            x = np.loadtxt(wav)
        except:
            x = range(hsi.shape[2])  # [st.parse2slice(args.slice)[2]]
    else:
        raise ValueError("Format \"%s\" is not supported." % filename.split(
            ".")[-1] + "\nSupported formats: float, pkl.")

    return hsi, x


def getparse():
    # command line arguments
    parser = argparse.ArgumentParser(description='Convert float file to image')
    parser.add_argument(dest='file_in', metavar='FLOAT')
    parser.add_argument('-o', '--out-suffix',
                        metavar='SUFFIX', type=str, default='conv', help='Specify output (ex. spec.png spec.txt DIR/out.png)')
    parser.add_argument('-r', '--ref', dest='file_ref',
                        default=None, metavar='REFERENCE')
    parser.add_argument('-w', dest='wavelength')
    parser.add_argument('-t', '--transmittance',
                        action='store_true', required=False, default=False)
    parser.add_argument('-a', '--absorbance',
                        action='store_true', required=False, default=False)
    parser.add_argument('-d1', '--div1',
                        action='store_true', required=False, default=False)
    parser.add_argument('-d2', '--div2',
                        action='store_true', required=False, default=False)
    parser.add_argument('-sm', '--smooth', metavar='KERNEL_SIZE',
                        type=float, required=False, default=3.0)
    parser.add_argument('-c', '--PseudoColor',
                        action='store_true', required=False, default=False)
    parser.add_argument('-m', '--movie',
                        action='store_true', required=False, default=False)
    parser.add_argument('-yx', '--ymax', metavar='YMAX',
                        type=float, required=False, default=None)
    parser.add_argument('-ym', '--ymin', metavar='YMIN',
                        type=float, required=False, default=None)
    parser.add_argument('-sl', '--slice', type=str,
                        required=False, default=":,:,:")
    return parser.parse_args()

args = getparse()

hsi, wav = get_hsi_wav(args.file_in, wav=args.wavelength)
hsi = hsi[st.parse2slice(args.slice)]
wav = wav[st.parse2slice(args.slice)[2]]

if (args.absorbance or args.transmittance) and args.file_ref == None:
    raise ValueError("Specify reference as -r.")
if args.file_ref and not(args.transmittance or args.absorbance):
    print("##WARNING: Reference file is not used.")

if args.file_ref:
    hsi_l, _ = get_hsi_wav(args.file_ref)

if args.absorbance:
    hsi = abs(hsi, hsi_l)
elif args.transmittance:
    hsi = trns(hsi, hsi_l)
elif args.div1:
    hsi = div1(x, hsi, 1, smooth=args.smooth)
elif args.div2 != 0:
    hsi = div1(x, hsi, 2, smooth=args.smooth)

minn, maxx = hsi.min(), hsi.max()

if args.transmittance:
    minn, maxx = 0., 1.
if args.absorbance:
    minn, maxx = 0., 2.
if args.ymin:
    minn = args.ymin
if args.ymax:
    maxx = args.ymax

try:
    wav = int(wav)
    wav = [wav]
    hsi = hsi[:, :, np.newaxis]
except:
    pass


for i, w in enumerate(wav):
    suff = os.path.splitext(args.out_suffix)[1]
    outf = os.path.splitext(args.out_suffix)[0]
    if suff == "":
        suff = ".png"
    file_out = st.mkdir_suff(outf, "conv", ex="_%04d" % w + suff)

    img = hsi[:, :, i]
    if suff == '.txt':
        np.savetxt(file_out, img)
    elif args.PseudoColor:
        socio.savefig(file_out, img, max=maxx, min=minn, color="c")
    elif suff == ".png":
        socio.savefig(file_out, img, max=maxx, min=minn)
    elif suff == ".pkl":
        file_out = args.out_suffix
        st.savepickle(file_out, hsi)

    print 'Saved to ' + file_out
    if suff == ".pkl":
        break

if args.movie:
    print("SocMakeMovie.py %s"%(outf + "*" + suff))
    os.system("SocMakeMovie.py %s"%(outf + "*" + suff))

