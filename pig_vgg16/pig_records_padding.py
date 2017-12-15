# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 23:21:19 2017

@author: DELL
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import os.path
import sys

#from PIL import Image
import cv2

import numpy as np
import tensorflow as tf
from tensorflow.python.platform import gfile
import config

IMAGE_HEIGHT = config.IMAGE_HEIGHT  #256
IMAGE_WIDTH = config.IMAGE_WIDTH   #256
CLASSES_NUM = config.CLASSES_NUM   #10


RECORD_DIR = config.RECORD_DIR
TRAIN_FILE = config.TRAIN_FILE
VALID_FILE = config.VALID_FILE

FLAGS = None

def _int64_feature(value):
  return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _bytes_feature(value):
  return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def label_to_one_hot(label):
  one_hot_label = np.zeros([CLASSES_NUM])
  one_hot_label[label] = 1.0
  return one_hot_label.astype(np.uint8)  #(4,10)


def conver_to_tfrecords(data_set, name):
  """Converts a dataset to tfrecords."""
  if not os.path.exists(RECORD_DIR):
      os.makedirs(RECORD_DIR)
  filename = os.path.join(RECORD_DIR, name)
  print('>> Writing', filename)
  writer = tf.python_io.TFRecordWriter(filename)
  data_set_list=list(data_set)
  num_examples = len(data_set_list)
  count = 0
  for index in range(num_examples):
    count += 1
    image = data_set_list[index][0]
    height = image.shape[0]
    width = image.shape[1]
    image_raw = image.tostring()
    label = data_set_list[index][1]
    label_raw = label_to_one_hot(label).tostring()
    example = tf.train.Example(features=tf.train.Features(feature={
        'height': _int64_feature(height),
        'width': _int64_feature(width),
        'label_raw': _bytes_feature(label_raw),
        'image_raw': _bytes_feature(image_raw)}))
    writer.write(example.SerializeToString())
    if count %500 == 0:
	    print('processed {}/{}'.format(count,num_examples))
  writer.close()
  print('>> Writing Done!')


def create_data_list(image_dir):
  if not gfile.Exists(image_dir):
    print("Image director '" + image_dir + "' not found.")
    return None
  extensions = [ '*.jpg']
  print("Looking for images in '" + image_dir + "'")
  file_list = []
  for extension in extensions:
    file_glob = os.path.join(image_dir, extension)
    file_list.extend(gfile.Glob(file_glob))
  if not file_list:
    print("No files found in '" + image_dir + "'")
    return None
  images = []
  labels = []
  all_list = len(file_list)
  count = 0
  for file_name in file_list:
    count += 1
    image = cv2.imread(file_name)
    h,w,c = image.shape
    fig = np.ones((IMAGE_WIDTH,IMAGE_WIDTH,c))*255
    if h>w:
        rate = h/IMAGE_WIDTH
        h_v = IMAGE_WIDTH
        w_v = int(w/rate)
        border = (IMAGE_WIDTH - w_v)
        up = 0
        down = IMAGE_WIDTH + 1
        left = int(border/2)
        right = int(IMAGE_WIDTH - border/2)
    else:
        rate = w/IMAGE_WIDTH
        w_v = IMAGE_WIDTH
        h_v = int(h/rate)
        border = (IMAGE_WIDTH - h_v)
        up = int(border/2)
        down = int(IMAGE_WIDTH - border/2)
        left = 0
        right = IMAGE_WIDTH + 1
            
    img_v = cv2.resize(image, (w_v, h_v))
    fig[up:down, left:right] = img_v
    input_img = np.array(fig, dtype='int16')
#    fig = fig.astype(img_v.dtype)
#    cv2.imwrite('q.jpg', input_img)
#    cv2.imwrite('p.jpg', fig)
    label_name = int(os.path.basename(file_name).split('_')[0]) - 1   #start at 0
    images.append(input_img)
    labels.append(label_name)
    if count % 500 == 0:
	    print('processed :{}/{}'.format(count,all_list))
  return zip(images, labels)


def main(_):
  training_data = create_data_list(FLAGS.train_dir)
  conver_to_tfrecords(training_data, TRAIN_FILE)


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--train_dir',
      type=str,
      default='pig_body/train_data',
      help='Directory training to get captcha data files and write the converted result.'
  )
  parser.add_argument(
      '--valid_dir',
      type=str,
      default='pig_body/valid_data',
      help='Directory validation to get captcha data files and write the converted result.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)


