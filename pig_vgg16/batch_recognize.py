from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os.path
from datetime import datetime
from PIL import Image
import numpy as np

import tensorflow as tf
from tensorflow.python.platform import gfile
import pig_model
import csv
import config

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT
CLASSES_NUM = config.CLASSES_NUM

FLAGS = None
Batch_size = 1

def input_data(image_dir):
  if not gfile.Exists(image_dir):
    print(">> Image director '" + image_dir + "' not found.")
    return None

  print(">> Looking for images in '" + image_dir + "'")


  file_glob = os.path.join(image_dir, '*.JPG' )
  file_list = gfile.Glob(file_glob)
  if not file_list:
    print(">> No files found in '" + image_dir + "'")
    return None
  file_list = sorted(file_list)
  all_files = len(file_list)
  images = np.zeros([all_files, IMAGE_HEIGHT*IMAGE_WIDTH*3], dtype='float32')
  files = []
  i = 0
  for file_name in file_list:
    image = Image.open(file_name)
    image_resize = image.resize(size=(IMAGE_WIDTH,IMAGE_HEIGHT))
    input_img = np.array(image_resize, dtype='float32')
    input_img = input_img.flatten()/127.5 - 1
    images[i,:] = input_img
    base_name = os.path.basename(file_name)
    files.append(base_name)
    i += 1
  return images, files

parser = argparse.ArgumentParser()
parser.add_argument(
    '--checkpoint_dir',
    type=str,
    default='checkpoint',
    help='Directory where to restore checkpoint.'
)
parser.add_argument(
    '--test_dir',
    type=str,
    default='pig_body/body_test',
    help='Directory where to get captcha images.'
)
FLAGS, unparsed = parser.parse_known_args()

with tf.Graph().as_default():
  input_images, input_filenames = input_data(FLAGS.test_dir)#得到文件夹内所有照片和文件名
  max_step = len(input_images)
  images = tf.placeholder(tf.float32,[IMAGE_HEIGHT*IMAGE_WIDTH*3],name ='input')
  logits = pig_model.inference(images, keep_prob=1,is_training=True)
  output = pig_model.predict(logits)
  saver = tf.train.Saver()
  sess = tf.Session()
  saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
  tag = []
  for each in range(max_step):
      feed_dict = input_images[each]
      recog_result = sess.run(output,feed_dict={images:feed_dict})
      tag.append(list(recog_result))
  np.save('predict_body',tag)
  sess.close()