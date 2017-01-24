#!/usr/bin/env python

import cv2
import numpy as np
import sys
import os

drawing = False
sx, sy = 0, 0
gx, gy = 0, 0
rectangles = []
count = 0
basefile = []
common = []
stackimg = []


def draw_circle(event, x, y, flags, param):
    global sx, sy, gx, gy, drawing, count, basefile

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        sx, sy = x, y

    elif event == cv2.EVENT_MOUSEMOVE:
        if x > 0 and x < img.shape[1]:
            gx = x
        if y > 0 and y < img.shape[0]:
            gy = y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False

        filename = basefile.split(".")[0] + "/cord" + "_%03d.txt" % count
        sx, x = np.min([x, sx]), np.max([x, sx])
        sy, y = np.min([y, sy]), np.max([y, sy])

        B = np.zeros(img_.shape[:2], img_.dtype)
        B[sy:y + 1, sx:x + 1] = 1
        common = mask_inv * B

        try:
            X = np.array(zip(*np.where(common != 0)))[:, ::-1]
            rectangles.append(X)
            np.savetxt(filename, X, fmt="%d")
            # print count,"(%d,%d),(%d,%d)\t"%(sx,sy,x,y), "savetxt...
            # %s"%filename
            print "savetxt... %s" % filename
            stackimg.append(drawedge(stackimg[-1], X, count))
            count += 1
        except:
            print "### MESSAGE : slect non-mask region"


cv2.namedWindow('image')
cv2.setMouseCallback('image', draw_circle)

# make directry
basefile = sys.argv[1]
if os.path.exists(basefile.split(".")[0]):
    pass
else:
    os.mkdir(basefile.split(".")[0])

# load raw and mask image
img_ = cv2.imread(sys.argv[1])
try:
    mask = cv2.imread(sys.argv[2])
except:
    mask = np.zeros(img_.shape, img_.dtype)


def screen(base, comp, alpha=0.2):
    max_value = float(np.iinfo(base.dtype).max)
    base_ = base.copy().astype(float)
    comp_ = comp.copy().astype(float) * alpha
    return (base_ + comp_ - (base_ * comp_) / max_value).astype("uint8")


def processmask(mask):
    mask[mask < np.iinfo(mask.dtype).max] = 0
    a = mask[:, :, 0]
    a = zip(*np.where(a == np.iinfo(a.dtype).max))
    for i in a:
        mask[i] = [255, 0, 0]
    mask_inv = (-mask + np.iinfo(mask.dtype).max)[:, :, 0]
    mask_inv[mask_inv != 0] == 1

    return mask, mask_inv


def drawedge(img, cordinates, j, color=[0, 255, 0]):
    img_result = img.copy()
    A = np.zeros(img.shape[:2], img.dtype)
    for i in cordinates[:, ::-1]:
        A[tuple(i)] = 255
    edge = cv2.Laplacian(A, cv2.CV_32F).astype("uint8")
    cord = zip(*np.where(edge != 0))
    for i in cord:
        img_result[i] = color
    cv2.putText(img_result, "%d" % j, (np.min(cordinates[:, 0]), np.min(
        cordinates[:, 1])), font, 1, (255, 255, 255), 2)
    cv2.putText(img_result, "%d" % j, (np.min(cordinates[:, 0]), np.min(
        cordinates[:, 1])), font, 1, (0, 0, 0), 1)
    return img_result


mask, mask_inv = processmask(mask)
img_ = screen(img_, mask)
stackimg.append(img_)

while True:
    img = stackimg[count].copy()

    if drawing:
        color = (0, 0, 255)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.rectangle(img, (sx, sy), (gx, gy), color, 1)
        cv2.putText(img, "(%d,%d),(%d,%d)" % (sx, sy, gx, gy),
                    (gx, gy), font, 1, (255, 255, 255), 2)
        cv2.putText(img, "(%d,%d),(%d,%d)" %
                    (sx, sy, gx, gy), (gx, gy), font, 1, (0, 0, 0), 1)

    cv2.imshow('image', img)

    k = cv2.waitKey(1) & 0xFF
    if k == ord('d'):
        if rectangles and stackimg and count > 0:
            rectangles.pop()
            stackimg.pop()
            count -= 1
            print "### MESSAGE : delete rectangles : %d" % count
        else:
            print "### ERROR : rectangles are not exit"
    elif k == ord('q'):
        cv2.imwrite(basefile.split(".")[0] + "/cord" + ".png", img)
        print basefile.split(".")[0] + "/cord" + ".png"
        sys.exit()
