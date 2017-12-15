from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import os.path
from datetime import datetime

import numpy as np
import cv2
import tensorflow as tf
from tensorflow.python.platform import gfile
import pig_model_dark as pig_model
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


  file_glob = os.path.join(image_dir, '*.jpg' )
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
    image = cv2.imread(file_name)
    input_img = np.array(image, dtype='float32')
    height, width, _ = input_img.shape
    offset = abs(height -width)//2
    if height >= width:
      pad_image = np.pad(input_img,((0,0),(offset,offset),(0,0)),mode='constant',constant_values =0)
    else:
      pad_image = np.pad(input_img,((offset,offset),(0,0),(0,0)),mode='constant',constant_values =0)
    
    image_resize = cv2.resize(pad_image, (IMAGE_WIDTH,IMAGE_HEIGHT))
#    cv2.imwrite('a.jpg', pad_image)
#    cv2.imwrite('b.jpg', image_resize)

    image_resize = image_resize.flatten()/127.5 - 1
    images[i,:] = image_resize
    base_name = os.path.basename(file_name)
    files.append(base_name)
    i += 1
  return images, files


def run_predict():
  with tf.Graph().as_default():
    input_images, input_filenames = input_data(FLAGS.test_dir)#得到文件夹内所有照片和文件名
    max_step = len(input_images)
    images = tf.placeholder(tf.float32,[IMAGE_HEIGHT*IMAGE_WIDTH*3],name ='input')
    logits = pig_model.inference(images, keep_prob=1,is_training=False)
    output = pig_model.output(logits)
    saver = tf.train.Saver()
    sess = tf.Session()
    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    tag = []
    for each in range(max_step):
        feed_dict = input_images[each]
        recog_result = sess.run(output,feed_dict={images:feed_dict})
        recog_result = np.squeeze(recog_result)
        current_name = str(int(input_filenames[each].split('.')[0]))
        for i in range(len(recog_result)):
          tag.append([current_name,str(i+1),str('%.8f'%(recog_result[i]))])
    with open('out.csv','w',newline='') as csvfile:
      writer = csv.writer(csvfile)
      for x in tag:
        writer.writerow(x)
    sess.close()
    return tag

def main(_):
  tag = run_predict()

if __name__ == '__main__':
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
      default='pig_body/train_data1',
      help='Directory where to get captcha images.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
