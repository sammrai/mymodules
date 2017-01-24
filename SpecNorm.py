#!/usr/bin/python

import numpy as np
import argparse
import re


def Energy(x, y):
    c = 299792458.
    h = 6.626e-34
    E = []
    for (i, j) in zip(x, y):
        E.append(j / i)
    return np.sum(E)


def surface_era(x, y):
    E = []
    dx = (x[1] - x[0])
    for (i, j) in zip(x, y):
        E.append(j * dx)
    return np.sum(E)


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


def findNearestWavelength(wavelength, spec):
    vec = spec[:, 0] - np.ones(len(spec[:, 0])) * wavelength
    vec = np.abs(vec)
    vec = vec.tolist()
    wl = vec.index(min(vec))
#    print 'i='+str(wl)+', lambda='+str(spec[wl,0])
    return wl


# Parse program arguments
parser = argparse.ArgumentParser(description='Convert spectrum')
parser.add_argument(dest='file_in', nargs='+', metavar='DATA')
parser.add_argument('-o', '--out-suffix', metavar='SUFFIX',
                    type=str, default='norm')
parser.add_argument('-w', '--wavelength', metavar='WAVELENGTH',
                    type=float, required=False, default=None)
parser.add_argument('-wx', '--wave-max', metavar='MAX_WAVELENGTH',
                    type=float, required=False, default=None)
parser.add_argument('-wm', '--wave-min', metavar='MIN_WAVELENGTH',
                    type=float, required=False, default=None)
parser.add_argument('-e', '--energy', action='store_true',
                    required=False, default=False)
parser.add_argument('-s', '--surface', action='store_true',
                    required=False, default=False)


args = parser.parse_args()


# parse files
spec_in = map(parseSpectrum, args.file_in)

# sakurai
if args.energy and args.surface:
    print "###ERROR : It should not be specified at the same -s and -e"
    exit()

elif args.energy:
    norm_spec = np.copy(spec_in)
    for i, spec in enumerate(norm_spec):
        E = Energy(spec[:, 0], spec[:, 1])
        norm_spec[i][:, 1] = spec[:, 1] / E

elif args.surface:
    norm_spec = np.copy(spec_in)
    for i, spec in enumerate(norm_spec):
        E = surface_era(spec[:, 0], spec[:, 1])
        norm_spec[i][:, 1] = spec[:, 1] / E

else:
    # normalizing average
    norm_spec = np.copy(spec_in)
    for i, spec in enumerate(norm_spec):
        max_index = spec[:, 1].tolist().index(max(spec[:, 1]))
    #    norm_spec[i][:,1] = spec[:,1] * len(spec[:,0]) / np.sum(spec[:,1])
        norm_spec[i][:, 1] = spec[:, 1] / spec[max_index, 1]


# save
for i, file_name in enumerate(args.file_in):
    file_out = file_name.split('/')[-1]
#    file_out = re.sub(r'_avg|_conv|\.[a-zA-Z]+', '', file_out)
    file_out = re.sub(r'_[a-zA-Z]+\.[a-zA-Z]+', '', file_out)
    file_out += '_' + args.out_suffix + '.txt'

    np.savetxt(file_out, norm_spec[i], fmt='%12.8f')
    print 'Saved to ' + file_out

# print 'lambda='+str(spec_in[0][best_wl,0])+' is selected,
# error='+str(err_min)+'\n'
print

exit()


# find the best data
sum_spec = [np.sum(spec[:, 1]) for spec in spec_in]
best_spec_index = sum_spec.index(max(sum_spec))
best_spec = np.copy(spec_in[best_spec_index])


norm_spec = []
err_min = float('inf')

wl_min = 0
wl_max = len(spec_in[0][:])

if args.wave_max is not None:
    wl_max = findNearestWavelength(args.wave_max, spec_in[0])

if args.wave_min is not None:
    wl_min = findNearestWavelength(args.wave_min, spec_in[0])

if args.wavelength is not None:
    wl_min = wl_max = findNearestWavelength(args.wavelength, spec_in[0])
    wl_max += 1

# find the best wavelength
coef = 0.0
best_wl = 0
for wl in range(wl_min, wl_max):
    err = 0.
    norm = spec_in[:]
    for i, spec in enumerate(spec_in):
        coef = best_spec[wl, 1] / spec[wl, 1]
        # coef = np.sum(best_spec[wl_min:wl_max+1,1]) / np.sum(spec[wl_min:wl_max+1,1])
        spec_n = np.copy(spec)
        spec_n[:, 1] = spec[:, 1] * coef
        norm[i] = spec_n
        err += np.linalg.norm(spec_n[wl_min:wl_max + 1, 1] -
                              best_spec[wl_min:wl_max + 1, 1])

#    avg = np.sum(norm[:])/len(norm)
#    err = np.linalg.norm(norm[:]-avg)

#    print str(spec_in[0][wl,0])+': '+ str(err)
    if err < err_min:
        err_min = err
        norm_spec = norm[:]
        best_wl = wl


# save
for i, file_name in enumerate(args.file_in):
    file_out = file_name.split('/')[-1]
    file_out = re.sub(r'_avg|_conv|\.[a-zA-Z]+', '', file_out)
    file_out += '_' + args.out_suffix + '.txt'

    np.savetxt(file_out, norm_spec[i], fmt='%12.8f')
    print 'Saved to ' + file_out

print 'lambda=' + str(spec_in[0][best_wl, 0]) + ' is selected, error=' + str(err_min) + '\n'
print
