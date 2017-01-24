#!/usr/bin/env python

import argparse
import os
import PIL.Image
import PIL.ImageDraw
import PIL.ImageFont
import shutil
import sys
FONT_TYPE = "Arial.ttf"
tempDIR = ".temp"

if os.system("type ffmpeg >/dev/null 2>&1 ") != 0:
    print "#ERROR ffmpeg is not exist."
    exit()

# command line arguments
parser = argparse.ArgumentParser(
    description='Convert image files to AVI with caption of wavelength')
parser.add_argument(dest='src', nargs='+', metavar='IMG')
parser.add_argument('-o', '--out', type=str, default='movie.mp4',
                    help='specify output (default: movie.mpg, suffix: mpg, avi, mov, wmv)')
parser.add_argument('-f', '--fps', type=int, default=15,
                    help='FPS (default: 15)')
parser.add_argument('-n', '--txt', action='store_true',
                    required=False, default=False, help="print filename")
args = parser.parse_args()


def get_progressbar_str(progress, message=""):
    MAX_LEN = 30
    BAR_LEN = int(MAX_LEN * progress)
    return ('[' + '=' * BAR_LEN +
            ('>' if BAR_LEN < MAX_LEN else '') +
            ' ' * (MAX_LEN - BAR_LEN) +
            '] %.1f%%' % (progress * 100.) + "\t" + message)


def drawtext(draw, pos, text, fontsize, outlinesize=1):
    fillcolor = "white"
    shadowcolor = "black"
    try:
        font = PIL.ImageFont.truetype(FONT_TYPE, size=fontsize)
    except:
        raise "#ERROR: font error."
    x, y = pos
    draw.text((x - outlinesize, y - outlinesize),
              text, font=font, fill=shadowcolor)
    draw.text((x + outlinesize, y - outlinesize),
              text, font=font, fill=shadowcolor)
    draw.text((x - outlinesize, y + outlinesize),
              text, font=font, fill=shadowcolor)
    draw.text((x + outlinesize, y + outlinesize),
              text, font=font, fill=shadowcolor)
    draw.text((x, y), text, font=font, fill=fillcolor)


try:
    os.mkdir(tempDIR)
except:
    # print  "#ERROR : temp folder is already exists. remove %s"%tempDIR
    shutil.rmtree(tempDIR)
    os.mkdir(tempDIR)


for i, i_ in enumerate(args.src):
    if args.txt:
        # print "processing... "+i_
        text = os.path.basename(i_)  # .split(".")[0]
        inputfile = i_
        outputfile = tempDIR + "/temp_%05d" % i + os.path.splitext(i_)[1]

        img = PIL.Image.open(i_)
        draw = PIL.ImageDraw.Draw(img)
        drawtext(draw, (10, 10), text, 30)
        img.save(outputfile)
    else:
        os.system("cp %s %s" % (i_, tempDIR + "/temp_%05d" %
                                i + os.path.splitext(i_)[1]))

    progress = (i / float(len(args.src) - 1))
    sys.stderr.write(
        '\r\033[K' + get_progressbar_str(progress, message="%s" % i_))
    sys.stderr.flush()
print


vcodec = "libx264"
pix_fmt = "yuv420p"
os.system("ffmpeg -r %d -i %s/temp_\%s.png -vcodec %s -pix_fmt %s %s -y >/dev/null 2>&1 " %
          (args.fps, tempDIR, "%05d", vcodec, pix_fmt, args.out))
shutil.rmtree(tempDIR)
print 'Saved to ' + args.out
