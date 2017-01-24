#!/usr/bin/env python

import numpy as np
import argparse
import re
import matplotlib.pyplot as plt
import os
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


def fl(lam, x):
    for i, l in enumerate(x):
        if l > lam:
            ind = i
            break
    return ind


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

# Parse program arguments
parser = argparse.ArgumentParser(description='Convert spectrum')
parser.add_argument(dest='file_in', nargs='+', metavar='SOURCE')
parser.add_argument('-o', '--out-suffix', metavar='SUFFIX',
                    type=str, default='org')
parser.add_argument('-av', '--average', action='store_true',
                    required=False, default=False)
parser.add_argument('-sl', '--slice', type=str,  required=False)
parser.add_argument('-w', '--wav', dest='file_wav',
                    default=None, metavar='wavelength')

args = parser.parse_args()


# parse files
spec_in = np.array(map(parseSpectrum, args.file_in))


if args.slice:
    spec_in = [i[(parse2slice(args.slice))] for i in spec_in]


spec_out = np.copy(spec_in[0])
spec_out[:, 1] = [0. for i in spec_out[:, 1]]
SUFFIX = "org"


if args.average:
    spec_out[:, 1] = np.sum([spec_out[:, 1] + i[:, 1]
                             for i in spec_in], axis=0) / len(spec_in)

    # save
    file_out = st.mkdir_suff(args.out_suffix, SUFFIX,
                             base=args.file_in[0], ex=".txt")
    np.savetxt(file_out, spec_out, fmt='%12.8f')
    print 'Saved to ' + file_out
