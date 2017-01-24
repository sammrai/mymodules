#!/usr/bin/env python

import numpy as np
import argparse
import re
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


def ope(N):
    return bool(re.compile("^[-+/*]?[-]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?$").match(N))


def parselabel(filename):
    file_spec = open(filename)

    spectrum = []
    for line_raw in file_spec.readlines():
        line = line_raw.strip()

        spec = line.split()
        if isd(spec[0]):
            spectrum.append(float(spec[0]))
    return np.array(spectrum)


def warning(S):
    return "\033[31m" + S + "\033[0m"

parser = argparse.ArgumentParser(description='Convert spectrum')
parser.add_argument(dest='file_in', nargs='+', metavar='SOURCE')
parser.add_argument('-o', '--out-suffix', metavar='SUFFIX',
                    type=str, default='tx')
parser.add_argument('-r', '--ref', dest='file_ref',
                    default=None, metavar='REFERENCE')
parser.add_argument('-a', '--absorbance', action='store_true',
                    required=False, default=False)
parser.add_argument('-t', '--transmission',
                    action='store_true', required=False, default=False)
parser.add_argument('-s', '--sub', type=float,
                    required=False, nargs='?', default=0.)

# parser.add_argument('-o', '--out-suffix', metavar='SUFFIX', type=str, default='rev')
args = parser.parse_args()


# parse files
spec_in = np.array(map(parseSpectrum, args.file_in))


if args.file_ref is not None:
    if ope(args.file_ref):
        N = args.file_ref
        OPERATOR = N[0]
        NUM = float(N[1:])
        flag = "OPEATE"
        print "OPERATEmode"
    elif len(np.array(parselabel(args.file_ref))) == len(spec_in):
        print (len(np.array(parselabel(args.file_ref))), len(spec_in)),
        brix = np.array(parselabel(args.file_ref))
        flag = "BRIX"
        print "BRIXmode"
    else:
        print (len(np.array(parselabel(args.file_ref))), len(spec_in)),
        spec_ref = parseSpectrum(args.file_ref)
        flag = "SPEC"
        print "SPECmode"
else:
    print warning("###ERROR Detect reference file.")
    print "  -r REFERENCE, --ref REFERENCE"
    exit()


for i, spec in enumerate(spec_in):

    spec_out = np.copy(spec)

    if flag is "BRIX":
        spec_out[:, 1] = spec[:, 1] / brix[i]

    if flag is "SPEC":
        if args.transmission:
            spec_out[:, 1:2] = spec[:, 1:2] / spec_ref[:, 1:2]

        elif args.absorbance:
            spec[:, 1:2] = np.abs(spec[:, 1:2])
            spec_ref[:, 1:2] = np.abs(spec_ref[:, 1:2])
            spec_out[:, 1:2] = spec[:, 1:2] / spec_ref[:, 1:2]
            spec_out[:, 1:2] = 1.0 / spec_out[:, 1:2]
            spec_out[:, 1:2] = np.log10(spec_out[:, 1:2])
        elif args.sub != 0:
            # spec_ref[:,1:2]=[(i)**args.pow for i in spec_ref[:,1:2]]
            s = args.sub
            if s is None:
                s = 1.
            spec_out[:, 1:2] = spec[:, 1:2] - s * spec_ref[:, 1:2]

        else:
            print warning("###ERROR Specify option")
            print "  -a, --absorbance\n  -t, --transmission\n  -s [SUB], --sub [SUB]"
            exit()
    if flag is "OPERATE":
        # spec_ref[:,1:2]=[(i)**args.pow for i in spec_ref[:,1:2]]
        if OPERATOR is "-":
            spec_out[:, 1:2] = spec[:, 1:2] - NUM
        if OPERATOR is "+":
            spec_out[:, 1:2] = spec[:, 1:2] + NUM
        if OPERATOR is "/":
            spec_out[:, 1:2] = spec[:, 1:2] / NUM
        if OPERATOR is "*":
            spec_out[:, 1:2] = spec[:, 1:2] * NUM

    # if args.sub != 0:
    #     # spec_ref[:,1:2]=[(i)**args.pow for i in spec_ref[:,1:2]]
    #     s=args.sub
    #     if s is None:
    #         s=1.
    #     spec_out[:,1:2] = spec[:,1:2] -  s * spec_ref[:,1:2]


# save
    file_out = st.mkdir_suff(args.out_suffix, "tx",
                             base=args.file_in[i], ex=".txt")
    np.savetxt(file_out, spec_out, fmt='%12.8f')
    print 'Saved to ' + file_out
    if flag is "BRIX":
        print brix[i]
    else:
        pass
