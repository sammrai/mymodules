#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import os
import glob
import argparse


def fl(lam,x):
	for i,l in enumerate(x):
		if l>lam:
			ind=i
			break
	return ind


parser = argparse.ArgumentParser(description='plot spectrum')
parser.add_argument(dest='file_in', nargs='+', metavar='DATA')
parser.add_argument('-x', '--x-slice', metavar='SLICEWAVELENGTH', type=float, required=True)
parser.add_argument('-r', '--ref', default=None, metavar='REFERENCE')


args = parser.parse_args()
files = args.file_in


wav=np.loadtxt(files[0]).T[0]

i_=args.x_slice
lon=[]
for file in files:
	ii= fl(i_,wav)
	lon01=np.loadtxt(file)
	lon.append(lon01[ii][1])

# brix=np.loadtxt("_label2.txt")



xm=min(brix)-(max(brix)-min(brix))/5.
xx=max(brix)+(max(brix)-min(brix))/5.


np.savetxt("%s_brix.txt"%(i_),np.c_[brix,lon])
os.system("	PlotSpec.py %s_brix.txt -s -lx 'Brix' -c rainbow -o %s_brix.png -ld -lt '^' -ly 'Count' -fs 22 -xm %lf -xx %lf"%(i_,i_,xm,xx))
os.system("rm %s_brix.txt"%i_)
exit()

