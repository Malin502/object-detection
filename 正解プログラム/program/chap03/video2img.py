#!/usr/bin/python
# -*- coding: sjis -*-

import cv2

image_dir='./image_dir/'
image_file='img_%s.png'
cap = cv2.VideoCapture('sample-video.mp4')

i = 0
interval = 6
length = 60
while(cap.isOpened()):
    flag, frame = cap.read()  
    if flag == False:  
            break
    if i == length*interval:
            break
    if i % interval == 0:    
       cv2.imwrite(image_dir+image_file % str(i).zfill(6), frame)
       print('Save', image_dir+image_file % str(i).zfill(6))
    i += 1 
cap.release()  
