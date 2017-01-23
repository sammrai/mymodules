#!/usr/bin/env python
#! -*- coding: utf-8 -*-

import cv2
import numpy as np
import sys

drawing = False
sx, sy = 0, 0 
gx, gy = 0, 0
rectangles = []
count=0
basefile=[]
def draw_circle(event,x,y,flags,param):
    global sx, sy, gx, gy, drawing,count,basefile

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        sx,sy = x,y

    elif event == cv2.EVENT_MOUSEMOVE:
        if x > 0 and x < img.shape[1]:
            gx = x
        if y > 0 and y < img.shape[0]:
            gy = y

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        
        filename=basefile.split(".")[0]+"/cord"+"_%03d.txt"%count
        sx,x=np.min([x,sx]),np.max([x,sx])
        sy,y=np.min([y,sy]),np.max([y,sy])

        A=[]
        for i in range(sx,x+1):
            for j in range( sy,y+1):
                A.append([i,j])

        np.savetxt(filename,A,fmt="%d")
        print(count,"(%d,%d),(%d,%d)\t"%(sx,sy,x,y), "savetxt... %s"%filename)
        count+=1
        rectangles.append([(sx, sy), (x, y)])


img = np.zeros((512, 512, 3), np.uint8)
cv2.namedWindow('image')
cv2.setMouseCallback('image',draw_circle)
import os
basefile=sys.argv[1]

if os.path.exists(basefile.split(".")[0]):
    print("#")
else:
    os.mkdir(basefile.split(".")[0])

    rectangles = []

while True:
    img = cv2.imread(sys.argv[1])
    for i,r in enumerate(rectangles):
        cv2.rectangle(img, r[0], r[1], (0,255,0), 1)
        cv2.putText(img,"%d"%i,r[0],font, 1,(255,255,255),2)
        cv2.putText(img,"%d"%i,r[0],font, 1,(0,0,0),1)

    if drawing:

        color = (0, 0, 255)
        cv2.rectangle(img, (sx,sy), (gx,gy), color, 1)
        font = cv2.FONT_HERSHEY_PLAIN
        cv2.putText(img,"(%d,%d),(%d,%d)"%(sx,sy,gx,gy),(gx,gy),font, 1,(255,255,255),2)
        cv2.putText(img,"(%d,%d),(%d,%d)"%(sx,sy,gx,gy),(gx,gy),font, 1,(0,0,0),1)

        
    cv2.imshow('image', img)
    
    k = cv2.waitKey(1) & 0xFF
    if k == ord('d'):
        if rectangles and count>0:
            rectangles.pop()
            count-=1
            print("### MESSAGE : delete rectangles : %d"%count)
        else:
            print("### ERROR : rectangles are not exit")
    elif k == ord('q'):
        cv2.imwrite(basefile.split(".")[0]+"/cord"+".png",img)
        print(basefile.split(".")[0]+"/cord"+".png")
        sys.exit()

