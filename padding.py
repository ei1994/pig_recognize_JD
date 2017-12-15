#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Nov 25 10:52:47 2017

@author: no1
"""

import numpy as np
import cv2
import scipy.misc as misc
import glob
import os
paths = glob.glob('pig_data_face/*.jpg')
count = 0
for path in paths:
  basename = os.path.basename(path)
  label = basename.split('_')[0]
#  new_path = os.path.join('D:/pig_recognize/pig_slim1/pig_data_face_padding',label)
#  if not os.path.exists(new_path):
#    os.mkdir(new_path)

  img = cv2.imread(path)
  height, width, _ = img.shape
  if height<200 or width<200:
    os.remove(path)
    continue
  offset = abs(height -width)//2
  if height >= width:
    pad_image = np.pad(img,((0,0),(offset, offset),(0,0)),mode='constant',constant_values =0)
  else:
    pad_image = np.pad(img,((offset, offset),(0,0),(0,0)),mode='constant',constant_values =0)

  cv2.imwrite(os.path.join('D:/pig_recognize/pig_slim1/pig_data_face_padding', basename), pad_image)
  
  count += 1
  if count %500 ==0:
    print('processed {}/{}'.format(count, len(paths)))
