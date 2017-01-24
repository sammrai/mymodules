#!/usr/bin/python

import numpy as np
import argparse
import glob
import sys
import re


def parseSpectrumMaya(filename):
    label_body_start = '>>>>>Begin Spectral Data<<<<<'
    label_body_end = '>>>>>End Spectral Data<<<<<'

    file_spec = open(filename)
    flag_body = False

    spectrum = []
    for line_raw in file_spec.readlines():
        line = line_raw.strip()
        if label_body_end in line:
            break
        if flag_body:
            spec = line.split('\t')
            spectrum.append([float(spec[0]), float(spec[1])])
        if label_body_start in line:
            flag_body = True
    return np.array(spectrum)


def parseSpectrumMayaDemo(filename):
    label_body_start = '>>>>>Begin Processed Spectral Data<<<<<'
    label_body_end = '>>>>>End Processed Spectral Data<<<<<'

    file_spec = open(filename)
    flag_body = False

    spectrum = []
    for line_raw in file_spec.readlines():
        line = line_raw.strip()
        if label_body_end in line:
            break
        if flag_body:
            spec = line.split('\t')
            spectrum.append([float(spec[0]), float(spec[1])])
        if label_body_start in line:
            flag_body = True
    return np.array(spectrum)


def parseSpectrumBW(filename):
    label_body_start = 'Pixel;Wavelength;Wavenumber;'
    col_wlength = 0
    col_dark_sub = 0

    file_spec = open(filename)

    # find label line
    line = ' '
    while line:
        line = file_spec.readline()
        if not line:
            break
        if label_body_start in line:
            labels = line.split(';')
            for i, label in enumerate(labels):
                if 'Wavelength' in label:
                    col_wlength = i
                if 'Dark Subtracted' in label:
                    col_dark_sub = i
            break

    spectrum = []
    # read spectrum
    while True:
        line = file_spec.readline()
        if not line:
            break
        data = line.split(';')
        if '   ' in data[col_wlength]:
            continue
        spectrum.append([float(data[col_wlength]), float(data[col_dark_sub])])
    return np.array(spectrum)


# Parse program arguments
parser = argparse.ArgumentParser(description='Average given spectral data')
parser.add_argument(dest='file_filter', metavar='input',
                    help='prefix/part of dataset (may contain wildcards)')
args = parser.parse_args()

# List files
file_list = glob.glob(args.file_filter)
if len(file_list) == 0:
    print 'no file matches'
    exit()
print str(len(file_list)) + ' files'
print args.file_filter

# determine file type
first_file = open(file_list[0])
for line_raw in first_file.readlines():
    if 'BWSpec' in line_raw:
        parseSpectrum = parseSpectrumBW
        break
    elif 'Begin Processed Spectral Data' in line_raw:
        parseSpectrum = parseSpectrumMayaDemo
        break
    elif 'Begin Spectral Data' in line_raw:
        parseSpectrum = parseSpectrumMaya
        break
else:
    print 'unsupported file\n'
    exit()


num_data = 0
spectrum_all = []
spec_sum = []

# parse all matching files
for filename in file_list:
    idx = re.search('([0-9]+)$', filename.split('.')[0])
    sys.stdout.write(idx.group(1) + ' ')
    sys.stdout.flush()

    spec = parseSpectrum(filename)

    if num_data == 0:
        spec_sum = np.copy(spec)
    else:
        spec_sum[:, 1:2] += spec[:, 1:2]
    num_data += 1
print

# average data
spec_sum[:, 1:2] /= num_data

# save
file_out = args.file_filter.split('/')[-1]
file_out = re.sub(r'\*|\!', '', file_out) + '_avg.txt'

np.savetxt(file_out, spec_sum, fmt='%12.8f')
print 'Saved to ' + file_out
print ''
