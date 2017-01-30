#!/usr/bin/env python
# coding:utf-8
"""

Examples
----------------------------
socio is class to import .flat file. This document introduce you how to use socio class. first, import library as following.

  >>> import socio

Load file to make socio instance.

  >>> soc_instance=socio.openfloat(filename)

Load value examples

  >>> img=soc_instance[::]
  >>> img=soc_instance[100:200,0:100,:]
  >>> img=soc_instance[:,0:100,:]
  >>> img=soc_instance[100:200,222,:]

Save image

  >>> img=soc_instance[:,:,50]
  >>> socio.savefig("filename.png",img)

"""

import numpy as np
import struct
import os.path
import sys
from scipy.misc import imsave
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
plt.switch_backend('agg')


def savefig(filename, img, max=None, min=None, color=False, message="", colormap='rainbow', position="right", ticks=None):
    """
    Parameters
    ----------
    Arguments:
        filename : str
            A string containing a path to a filename
        img : 2-d array
            set 2d array

    Keyword arguments:
        max(min) : max(min) value
            If max(min) is not specified, value will be set automatucally.
        color : None or "c"
            If color is not specified, image is saved by glayscale color."c" option save image as pseudcoolor
        message :  str
            Display any message on top of the image. This argument is only available when color option is "c".

    """
    if img.ndim != 2:
        raise Exception("#ERROR : dimension number is invarid.")
    if min is not None and max is not None and min > max:
        raise Exception("#ERROR : min is bigger than max.")

    img = np.array(img)
    img = np.array([0 if "nan" == str(i) else i for i in list(
        img.reshape(-1))]).reshape(img.shape)
    if min is None:
        min = np.min(img)
    if max is None:
        max = np.max(img - min)
    if ticks:
        ticks=np.arange(min,max+(np.abs(max-min)*1e-5),ticks)
    if position == "bottom" or position == "top":
        orientation = "horizontal"
    else:
        orientation = "vertical"

    if not color:
        img -= min
        img /= max
        img = img * 65535.
        imsave(filename, img.astype(np.uint16))
    else:
        plt.clf()
        imgplot = plt.imshow(img, interpolation='none')
        imgplot.set_cmap(colormap)
        imgplot.set_clim(vmin=min, vmax=max)

        # colorbar
        ax = plt.gca()
        ax.set_axis_off()
        ax.set_title(message)
        divider = make_axes_locatable(ax)
        cax = divider.append_axes(position, size="5%", pad=0.05)
        cb = plt.colorbar(imgplot, cax=cax,ticks=ticks,orientation=orientation) 

        # plot
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

def get_progressbar_str(progress, message=""):
    MAX_LEN = 50
    BAR_LEN = int(MAX_LEN * progress)
    return (message[0:5] + '[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) +
            '] %.1f%%' % (progress * 100.))


class socio:

    def __init__(self):
        self.bpp = 4
        self.sample_cache = {}
        self.band_cache = {}
        self.line_cache = {}
        self.raw = None

    def getWavelengths(self):
        return self.wavelengths

    def openfloat(self, file_name):
        # file name
        fname_base = file_name.split('.float')[0]
        self.fname_header = fname_base + '.hdr'
        self.fname_float = fname_base + '.float'

        # check file exists
        if not os.path.exists(self.fname_header):
            print 'Missing header file', self.fname_header
            exit()
        if not os.path.exists(self.fname_float):
            print 'Missing float file'
            exit()

        # parse header file
        flag_wl_line = False
        for line in open(self.fname_header):
            if flag_wl_line:
                self.wavelengths = map(float, line.split(',')[:-1])
                self.wavelengths = np.array(self.wavelengths)

            if 'samples =' in line:
                self.samples = int(line.split()[-1])

            if 'lines   =' in line:
                self.lines = int(line.split()[-1])

            if 'bands   =' in line:
                self.bands = int(line.split()[-1])

            if 'header offset =' in line:
                self.header_offset = int(line.split()[-1])

            if 'wavelength =' in line:
                flag_wl_line = True

    def getSpectrumAt(self, point):
        (x, y) = point
        array = self.getSampleArray(x)
        return array[y, :]

    def getBandArray(self, band):
        # check cache
        if band in self.band_cache:
            return self.band_cache[band]
        # parse float file
        file_float = open(self.fname_float, 'rb')

        spec_array = np.zeros((self.lines, self.samples))
        file_float.seek(self.header_offset, 0)  # return to top of data
        file_float.seek(band * self.samples * self.bpp,
                        1)  # offset of each band
        for line in range(self.lines):
            for i in range(self.samples):
                val = file_float.read(self.bpp)
                val = struct.unpack('f', val)[0]
                spec_array[line, i] = val
            file_float.seek((self.bands - 1) * self.samples * self.bpp, 1)
        file_float.close()
        self.band_cache[band] = spec_array
        return spec_array

    def getRaw(self):
        if self.raw is not None:
            return self.raw
        file_float = open(self.fname_float, 'rb')
        spec_array = np.zeros((self.lines * self.samples * self.bands))
        file_float.seek(self.header_offset, 0)  # return to top of data

        for j in range(self.bands):
            progress = (j / float(self.bands - 1))
            sys.stderr.write(
                '\r\033[K' + get_progressbar_str(progress, message="getArray"))
            sys.stderr.flush()

            for i in range((self.lines * self.samples)):
                val = file_float.read(self.bpp)
                val = struct.unpack('f', val)[0]
                spec_array[self.lines * self.samples * j + i] = val
                # print self.lines*self.samples*j+i
        print
        file_float.close()
        return spec_array.reshape((self.lines, self.bands, self.samples)).transpose(0, 2, 1)

    def getLineArray(self, line):
        # check cache
        if line in self.line_cache:
            return self.line_cache[line]

        # parse float file
        file_float = open(self.fname_float, 'rb')

        spec_array = np.zeros((self.bands, self.samples))
        file_float.seek(self.header_offset, 0)  # return to top of data
        file_float.seek(self.bands * self.samples *
                        self.bpp, 1)  # offset of each band
        for band in range(self.bands):
            for i in range(self.samples):
                val = file_float.read(self.bpp)
                val = struct.unpack('f', val)[0]
                spec_array[band, i] = val
        file_float.close()
        self.line_cache[line] = spec_array
        return spec_array

    def getSampleArray(self, sample):
        # check cache
        if sample in self.sample_cache:
            return self.sample_cache[sample]

        # parse float file
        file_float = open(self.fname_float, 'rb')

        spec_array = np.zeros((self.lines, self.bands))
        file_float.seek(self.header_offset, 0)  # return to top of data
        file_float.seek(sample * self.bpp, 1)
        for line in range(self.lines):
            for band in range(self.bands):
                val = file_float.read(self.bpp)
                val = struct.unpack('f', val)[0]
                spec_array[line, band] = val
                file_float.seek((self.samples - 1) * self.bpp, 1)
        file_float.close()
        self.sample_cache[sample] = spec_array
        return spec_array

    def digit(self, a):
        try:
            int(a)
            return True
        except:
            return False

    def __getitem__(self, key):
        # print key
        lines = range(0, self.lines)[key[0]]
        samples = range(0, self.samples)[key[1]]
        bands = range(0, self.bands)[key[2]]

        if self.digit(lines) and self.digit(samples):
            return self.getSpectrumAt((lines, samples))[key[2]]
        if self.digit(lines) and not self.digit(samples):
            return self.getLineArray(lines).T[key[1]][:, key[2]]
        if not self.digit(lines) and self.digit(samples):
            return self.getSampleArray(samples)[key[0]][:, key[2]]
        if key[0].start is None and key[0].stop is None and key[1].start is None and key[1].stop is None:
            if self.digit(bands):
                return self.getBandArray(bands)
            return self.getRaw()[:, :, key[2]]

        A = []
        for j in lines:
            for i in samples:
                # print    len(self.sample_cache), len( self.band_cache ) ,len(
                # self.line_cache)
                A.append(self.getSpectrumAt((i, j)))
                progress = (len(self.sample_cache)) / float(len(samples))
                sys.stderr.write(
                    '\r\033[K' + get_progressbar_str(progress, message="getSpectrumAt"))
                sys.stderr.flush()
        print
        return np.array(A).reshape(len(lines), len(samples), -1)[:, :, key[2]]
        # return np.array([self.getSpectrumAt((j,i)) for i in samples for j in
        # lines]).reshape(len(lines),len(samples),-1)

    def shape(self):
        return self.lines, self.samples, self.bands


def openfloat(file_name):
    instance = socio()
    instance.openfloat(file_name)
    return instance
