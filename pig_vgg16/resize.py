# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 22:38:02 2017

@author: DELL
"""
'''
resize images
'''

import cv2
import numpy as np
import glob
import os

size = 320
files = glob.glob('D:/pig_recognize_body/pig_body/train_data/*.jpg')
result_dir = 'pig_body/train_data_resize'

try:
     os.makedirs(result_dir)
except os.error:
     pass
for i in files:
        img = cv2.imread(i)
#        img = cv2.imread('D:/pig_recognize_body/pig_body/train_data/2_00001_pig.jpg')
        base_name = os.path.basename(i)
        
        h,w,c = img.shape
        fig = np.ones((size,size,3))*255
        
        if h>w:
            rate = h/size
            h_v = size
            w_v = int(w/rate)
            border = (size - w_v)
            up = 0
            down = size + 1
            left = int(border/2)
            right = int(size - border/2)
        else:
            rate = w/size
            w_v = size
            h_v = int(h/rate)
            border = (size - h_v)
            up = int(border/2)
            down = int(size - border/2)
            left = 0
            right = size + 1
                
        img_v = cv2.resize(img, (w_v, h_v))
        fig[up:down, left:right] = img_v
        cv2.imwrite(os.path.join(result_dir,base_name), fig)

#fig = fig.astype(img_v.dtype)
#cv2.waitKey(0)
#cv2.imshow('img.jpg', img_v)
#cv2.imshow('img1.jpg', fig)
#cv2.waitKey(0)
#cv2.waitKey(0)
#cv2.waitKey(0)
#cv2.waitKey(0)

