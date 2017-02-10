#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import re
import os
import argparse
import code
import st
import glob

import matplotlib.pyplot as plt
import matplotlib.cm as cm
import matplotlib.colors as colors
from matplotlib.font_manager import FontProperties
from matplotlib import rc
plt.switch_backend('agg')


def isd(N):
    return bool(re.compile("^[-+]?([0-9]+(\.[0-9]*)?|\.[0-9]+)([eE][-+]?[0-9]+)?$|nan$|^[-+]?inf$").match(N))


def isunicode(strings):
    try:
        strings = unicode(strings, 'utf-8')
    except:
        return False
    for i in strings:
        if (ord(i)) > 255:
            return True
    return False

# def spec(ls,ind):
#     N=len(ls)-1
#     dx=1./float(N)
#     pi=np.pi

#     x=ind
#     r = (np.sin(1.5*pi*x+pi+pi/4 + pi) + 1)/2.0
#     g = (np.sin(1.5*pi*x+pi+pi/4 + pi/2) + 1)/2.0
#     b = (np.sin(1.5*pi*x+pi+pi/4) + 1)/2.0
#     return b,g,r


def label_to_float(label):
    label = [str(l) for l in label]
    r = re.compile("[0-9]+\.[0-9]*|[0-9]+")
    A = [r.findall(l) for l in label]
    A = [float(i[-1]) for i in A]
    return A


def colormake(lis, color="jet", max=1.0, min=0.):
    lis = label_to_float(lis)
    if len(lis) != 1:
        lis = [(i - np.min(lis)) / (np.max(lis) - np.min(lis))
               * (len(lis) - 1) for i in lis]
    else:
        pass

    # 3/14 added
    if len(lis) != 1:
        lis = lis / np.max(lis)

    color_map = plt.get_cmap(color)
    return [color_map(x) for x in lis]
    ##

    # color_max=max
    # color_min=min
    # color_range = color_max - color_min
    # color_width = color_range / float(len(lis))
    # color_index_offset = math.ceil(color_min/color_width)
    # color_grade_num = len(lis) + color_index_offset + math.ceil((1.0-color_max)/color_width)
    # color_map = plt.get_cmap(color)
    # cmap_norm = colors.Normalize(0, color_grade_num-1)
    # color_smap = cm.ScalarMappable(norm=cmap_norm, cmap=color_map)
    # A=[]
    # for i in lis:
    #     line_color = color_smap.to_rgba(i+color_index_offset)
    #     A.append(line_color)
    # return A


def getparse():
    # Parse program arguments
    parser = argparse.ArgumentParser(description='plot spectrum')
    parser.add_argument(dest='file_in', nargs='+', metavar='spec_file')
    parser.add_argument('-o', '--out-suffix',
                        metavar='SUFFIX', type=str, default='spec')
    # parser.add_argument('-s', '--silent', action='store_true', required=False, default=False, help='do not show a preview')
    parser.add_argument('-ld', '--legend', action='store_true',
                        required=False, default=False)
    parser.add_argument('-ldo', '--legend-out',
                        action='store_true', required=False, default=False)
    parser.add_argument('-ldb', '--legend-best',
                        action='store_true', required=False, default=False)
    parser.add_argument('-ldh', '--legend-holizon',
                        action='store_true', required=False, default=False)
    parser.add_argument('-lf', '--label-file', type=str,
                        required=False, default=None)

    parser.add_argument('-ly', '--label-y', type=str,
                        required=False, default='Count')
    parser.add_argument('-lx', '--label-x', type=str, required=False,
                        default="lambda", help="Tex notation : '$\mathit{\lambda}$'")
    parser.add_argument('-lc', '--legendcolor', action='store_true',
                        required=False, default=False, help='set color based on legend quantity')
    parser.add_argument('-sl', '--sortlegend', action='store_true',
                        required=False, default=False, help='sort legend')
    parser.add_argument('-sl*', '--sortlegend_inv', action='store_true',
                        required=False, default=False, help='sort legend with inverse')

    parser.add_argument('-logy', '--log_y', action='store_true',
                        required=False, default=False)
    parser.add_argument('-logx', '--log_x', action='store_true',
                        required=False, default=False)
    parser.add_argument('-sn', '--sci', action='store_true',
                        required=False, default=False, help='use 10^e format on y axis')
    parser.add_argument('-s', '--seaborn', action='store_true',
                        required=False, default=False, help='Use seaborn')
    parser.add_argument('-g', '--ggplot', action='store_true',
                        required=False, default=False, help='Use seaborn')
    parser.add_argument('-b', '--bar', action='store_true',
                        required=False, default=False, help='bar graph')

    parser.add_argument('-lw', '--line-width', metavar='LINE_WIDTH',
                        type=float, required=False, default=1.0)
    parser.add_argument('-lt', '--line-type', metavar='LINE_TYPE',
                        type=str, required=False, default="-")
    parser.add_argument('-fi', '--fill', action='store_true',
                        required=False, default=False)
    parser.add_argument('-ms', '--marker-size', metavar='MARKER_SIZE',
                        type=float, required=False, default=6.0)
    parser.add_argument('-ds', '--datasize', metavar='DATA_SIZE', type=int,
                        required=False, default=None, help="Specify the number of plot data.")

    parser.add_argument('-as', '--aspect', metavar='ASPECT_RATIO',
                        type=float, required=False, default=1.0)
    parser.add_argument('-xm', '--xmin', metavar='XMIN',
                        type=float, required=False, default=None)
    parser.add_argument('-xx', '--xmax', metavar='XMAX',
                        type=float, required=False, default=None)
    parser.add_argument('-xt', '--xtick', metavar='TICK_INTERVAL',
                        type=float, required=False, default=None)
    parser.add_argument('-ym', '--ymin', metavar='YMIN',
                        type=float, required=False, default=None)
    parser.add_argument('-yx', '--ymax', metavar='YMAX',
                        type=float, required=False, default=None)
    parser.add_argument('-yt', '--ytick', metavar='TICK_INTERVAL',
                        type=float, required=False, default=None)
    parser.add_argument('-cc', '--line-color', metavar='LINE_COLOR',
                        type=str, required=False, default=None)
    parser.add_argument('-c', '--color', metavar='COLOR_MAP', type=str, required=False, default='rainbow',
                        help="examples:Accent,afmhot,BrBG,gist_earth,autumn,Blues,bone,brg,BuGn,BuPu,bwr,CMRmap,cool,coolwarm,copper,cubehelix,Dark2,flag,gist_heat,gist_ncar,gist_rainbow,gist_stern,GnBu,gnuplot,gnuplot2,gray,Greens,Greys,hot,hsv,jet,nipy_spectral,ocean,Oranges,OrRd,Paired,Pastel1,Pastel2,pink,PiYG,PRGn,prism,PuBu,PuBuGn,PuOr,PuRd,Purples,rainbow,RdBu,RdGy,RdPu,RdYlBu,RdYlGn,Reds,seismic,Set1,Set2,Set3,Spectral,spring,summer,terrain,winter,YlGn,YlGnBu,YlOrBr,YlOrRd")  # jet

    parser.add_argument('-cm', '--color-min', metavar='COLOR_MIN', type=float, required=False,
                        default=0.0, help='set lower limit of color mapping (default: 0.0)')
    parser.add_argument('-cx', '--color-max', metavar='COLOR_MAX', type=float, required=False,
                        default=1.0, help='set upper limit of color mapping (default: 1.0)')
    parser.add_argument('--dpi', metavar='DPI',  type=int,
                        required=False, default=150)

    parser.add_argument('-fs', '--FONTSIZE', metavar='FONTSIZE',
                        type=float, required=False, default=None)
    parser.add_argument('-fo', '--FONT', metavar='FONT',
                        type=str, required=False, default=None)
    parser.add_argument('-t', '--TITLE', metavar='TITLE',
                        type=str, required=False, default=None)
    parser.add_argument('-i', '--intaractive', action='store_true',
                        required=False, default=False, help="intaractive mode")

    # easy plot transportaion
    parser.add_argument('-w', dest='wavelength')

    return parser.parse_args()

args = getparse()

# global setting
major_line_width = 1.5


rc('mathtext', fontset='stixsans')
plt.rcParams['font.family'] = 'Times New Roman'

if args.seaborn:
    # print "###"
    try:
        import seaborn
    except:
        print("##WORNING: No module named seaborn\nUse ggplot instead")
        plt.style.use('ggplot')

if args.ggplot:
    plt.style.use('ggplot')


# set label
if args.label_file is None:
    label_list = [fname.split('/')[-1] for fname in args.file_in]
    suff = "." + [fname.split('.')[-1] for fname in label_list][0]
    label_list = [s.strip(suff) for s in label_list]

else:
    label_file = open(args.label_file)
    label_list = label_file.readlines()
    label_list = [label.strip() for label in label_list]

# font settings
UNICODE_FLAG = isunicode(args.label_x) or isunicode(args.label_y) or isunicode(
    args.TITLE) or any([isunicode(i) for i in label_list])

if UNICODE_FLAG:
    # unicode font settings
    import sys
    reload(sys)
    sys.setdefaultencoding('utf-8')
    dirlist = ["/Library/Fonts/", "/System/Library/Fonts", ]
    target = ["*ipag*", "*Unicode*"]
    font_paths = [glob.glob(dirlist[i] + target[j])[0] for i in range(len(dirlist))
                  for j in range(len(target)) if glob.glob(dirlist[i] + target[j])]
    if not font_paths:
        print("##ERROR: japanese fonts is not found.")
        exit()

    font_path = font_paths[0]
    font_prop = FontProperties(fname=font_path)
    plt.rcParams['font.family'] = font_prop.get_name()
    plt.rcParams['pdf.fonttype'] = 42
    plt.rcParams['savefig.dpi'] = 300
elif args.FONT is not None:
    plt.rcParams['font.family'] = args.FONT

if args.FONTSIZE is not None:
    plt.rcParams['font.size'] = args.FONTSIZE

# parse files and check
file_in_ex_check = [os.path.splitext(i)[1] for i in args.file_in]
if not all(file_in_ex_check[0] == i for i in file_in_ex_check):
    print "##ERROR: invalid input extension."
    print file_in_ex_check
    exit()

spec_in = np.array(map(np.loadtxt, args.file_in))

# when the file num is 1
if spec_in.shape[0] == 1 and spec_in[0].shape[1] != 2:
    if args.wavelength:
        wav = list(np.loadtxt(args.wavelength))
    else:
        wav = range(spec_in.shape[2])

    wav = np.array(wav * spec_in.shape[1]).reshape(spec_in.shape[1:])
    spec_in = np.array([wav, spec_in[0]]).transpose(1, 2, 0)

if len(label_list) != len(spec_in):
    print "##ERROR: label_list is invalid."
    print "label length is", len(label_list), "spec_in length is", len(spec_in)
    exit()
    
# plot index
if args.datasize == None:
    args.datasize = len(spec_in)
plotlist = np.linspace(0, len(spec_in), args.datasize,
                       dtype=int, endpoint=False)
spec_in = spec_in[plotlist]

print len(spec_in)

# set aspect
fig = plt.figure()
ax = plt.subplot(1, 1, 1)
ax.set_aspect(1)
ax.set_axis_off()
w, h = fig.get_figwidth(), fig.get_figheight()
w = w / args.aspect
ax = plt.axes((0.52 - 0.5 * 0.8 * h / w, 0.12, 0.8 * h / w, 0.8))


if args.sortlegend:
    label_ind = np.argsort(np.array(
        [float(j) + float(i) * 1e-10 for i, j in enumerate(label_to_float(label_list))]))
    if args.sortlegend_inv:
        label_ind = np.argsort(AAA)[-1::-1]
    spec_in = [spec_in[i] for i in label_ind]
    label_list = [label_list[i] for i in label_ind]


# set color
if args.legendcolor:
    colorlist = colormake(label_list, color=args.color)
else:
    A = [i for i in range(len(spec_in))]
    colorlist = colormake(A, max=args.color_max,
                          min=args.color_min, color=args.color)


for i, spec in enumerate(spec_in):
    # if not i in plotlist: continue
    try:
        spec[:, 0], spec[:, 1] = [spec[j, 0] for j in np.argsort(
            spec[:, 0])], [spec[j, 1] for j in np.argsort(spec[:, 0])]
    except:
        pass

    line = args.line_width
    line_color = colorlist[i]

    if (args.line_color is not None):
        line_color = args.line_color
    if args.fill:
        zero = np.array([0. for i in range(len(spec_in[0]))])
        plt.fill_between(spec[:, 0], zero, spec[
                         :, 1], facecolor=line_color, linewidth=line, color=line_color, alpha=0.3)
    elif args.bar:
        try:
            plt.bar(spec[:, 0], spec[:, 1])
        except:
            plt.bar(range(len(spec)), spec)
    else:
        # plt.plot(spec[:,0],spec[:,1], args.line_type,linewidth=line, color=line_color, label=label_list[i],markersize=args.marker_size)
        try:
            if spec.shape[0] < spec.shape[1]:
                spec = spec.T
            plt.plot(spec[:, 0], spec[:, 1], args.line_type, linewidth=line,
                     color=line_color, label=label_list[i], markersize=args.marker_size)
        except:
            try:
                plt.plot(spec, args.line_type, linewidth=line, color=line_color,
                         label=label_list[i], markersize=args.marker_size)
            except:
                plt.plot(spec.T[0], spec.T[1], args.line_type, linewidth=line,
                         color=line_color, label=label_list, markersize=args.marker_size)

# set limitation
if args.xmin or args.xmin == 0.:
    plt.xlim(xmin=args.xmin)
if args.xmax or args.xmax == 0.:
    plt.xlim(xmax=args.xmax)
if args.ymin or args.ymin == 0.:
    plt.ylim(ymin=args.ymin)
if args.ymax or args.ymax == 0.:
    plt.ylim(ymax=args.ymax)
if args.xtick or args.xtick == 0.:
    plt.xticks(np.arange(ax.get_xlim()[0],ax.get_xlim()[1]+args.xtick,args.xtick))
if args.ytick or args.ytick == 0.:
    plt.yticks(np.arange(ax.get_ylim()[0],ax.get_ylim()[1]+args.ytick,args.ytick))


axissize = None
plt.xticks(fontsize=axissize)
plt.yticks(fontsize=axissize)

plt.ticklabel_format(style='sci', axis='x', scilimits=(-100, 100))
plt.ticklabel_format(style='sci', axis='y', scilimits=(-100, 100))
ax.xaxis.major.formatter._useMathText = True
ax.yaxis.major.formatter._useMathText = True
if args.sci:
    plt.ticklabel_format(style='sci', axis='y', scilimits=(0, 0))

fsize = min(230 / len(spec_in), axissize)
# legend
if args.legend:
    ax.legend(bbox_to_anchor=(1.0, 1.0), fontsize=fsize)
if args.legend_out:
    ax.legend(bbox_to_anchor=(1.0, 1.01), loc='upper left', fontsize=fsize)
if args.legend_best:
    ax.legend(fontsize=fsize, loc="best")
if args.legend_holizon:
    ax.legend(loc='best', ncol=len(label_list) / 2, fontsize=fsize)

# log scale
if args.log_y:
    plt.yscale('log')
if args.log_x:
    plt.xscale('log')

# label
if args.label_x != "None":
    if args.label_x is not "lambda":
        plt.xlabel(unicode(args.label_x, 'utf-8'), fontsize=fsize)
    else:
        plt.xlabel(r'$\mathit{\lambda}$ [nm]', fontsize=fsize)


if args.label_y != "None":
    plt.ylabel(unicode(args.label_y, 'utf-8'), fontsize=fsize)

if args.TITLE is not None:
    plt.title(unicode(args.TITLE, 'utf-8'))

if args.intaractive:
    code.InteractiveConsole(globals()).interact()

# plt.tight_layout()

# save
file_out = st.mkdir_suff(args.out_suffix, "conv",
                         base=args.file_in[0], ex=".pdf")
print 'Saved to ' + file_out


plt.savefig(file_out, bbox_inches='tight', pad_inches=0.2, dpi=args.dpi)
