#!/usr/bin/env python

import numpy as np
import argparse
import st


def getparse():
    # Parse program arguments
    parser = argparse.ArgumentParser(description='merge hsi.pkl data')
    parser.add_argument(dest='file_in', nargs='+', metavar='HSI_.pkl_file')
    parser.add_argument('-o', '--out-suffix',
                        metavar='SUFFIX', type=str, default='mrge')
    return parser.parse_args()


def merge(img1, img2):
    imgt = img1.dtype
    s1 = img1.shape
    s2 = img2.shape
    s_new = [s1[0] + s2[0], max(s1[1], s2[1])]
    newimg = np.array([[0. for i in range(s_new[1])] for j in range(s_new[0])])
    newimg[0:s1[0], 0:s1[1]] = img1
    newimg[s1[0]:s1[0] + s2[0], 0:s2[1]] = img2
    return newimg.astype(imgt)


def mergehsi2(hsi1, hsi2):
    return np.array([merge(hsi1[:, :, i], hsi2[:, :, i]) for i in range(hsi1.shape[2])]).transpose(1, 2, 0)


def mergehsi(hsis):
    mergeimage = hsis[0]
    for i in range(1, len(hsis)):
        mergeimage = mergehsi2(mergeimage, hsis[i])
    return np.array(mergeimage)


args = getparse()
fs = args.file_in
As = map(st.loadpickle, fs)
file_out = st.mkdir_suff(args.out_suffix, "mrge",
                         base=args.file_in[0], ex=".pkl")
print 'Saved to ' + file_out
st.savepickle(file_out, mergehsi(As))
