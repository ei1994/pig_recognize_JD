from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import sys
import tensorflow as tf
import pig_model as captcha
import logging
import numpy as np
import glob
import matplotlib.pylab as plt
import math
import config
import os
from datetime import datetime
'''
一头一头猪的数据进行训练
'''
learning_rate = 2e-4
epoch = 100
batch = 16
class_num = 30
FLAGS = None
IMAGE_WIDTH = 320
IMAGE_HEIGHT = config.IMAGE_HEIGHT

checkpoint_dir = 'ckpt'
checkpoint_file = os.path.join(checkpoint_dir, 'model.ckpt')
train_dir='summary2'

#def initLogging(logFilename='record.log'):#  """Init for logging
#  """
#  logging.basicConfig(
#                    level    = logging.DEBUG,
#                    format='%(asctime)s-%(levelname)s-%(message)s',
#                    datefmt  = '%y-%m-%d %H:%M',
#                    filename = logFilename,
#                    filemode = 'w');
#  console = logging.StreamHandler()
#  console.setLevel(logging.INFO)
#  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
#  console.setFormatter(formatter)
#  logging.getLogger('').addHandler(console)
#initLogging()

def inputs(epoch,batch,img_all,lab_all):
    for i in range(epoch):
        shuf_num = np.random.permutation(1)
#        shuf_num = 0
        for i in shuf_num:
            lab_all1 = lab_all[shuf_num[i]]                            
            img_all1 = img_all[shuf_num[i]] 
            for imgs in img_all1:
                batch_per_epoch = math.ceil(imgs.shape[0] / batch)
                for b in range(batch_per_epoch):
                    if (b*batch+batch)>imgs.shape[0]:
                        m,n = b*batch, imgs.shape[0]
                    else:
                        m,n = b*batch, b*batch+batch
        
                    x_batch, label_batch = img_all1[m:n,:],  lab_all1[m:n,:]
                    yield x_batch, label_batch
                
def label_to_one_hot(label):
  one_hot_label = np.zeros([1,class_num])
  one_hot_label[:,label] = 1.0
  return one_hot_label.astype(np.uint8)  #(4,10)    


def run_train():
  """Train CAPTCHA for a number of steps."""
  lab_all = []
  img_all = []
  for i in range(1):
      file = glob.glob('D:/pig_recognize_body/pig_body/train_data/' + str(i+1) + '_*.jpg')
      images = plt.imread(file[0])
      images = plt.resize(images, [IMAGE_WIDTH,IMAGE_WIDTH,3])
      images = np.reshape(images, [-1,IMAGE_WIDTH*IMAGE_WIDTH*3])
#      images = np.expand_dims(images,0)
#      num = len(file)
      m = 1
      for j in file:
          image = plt.imread(j)
          image = plt.resize(image, [IMAGE_WIDTH,IMAGE_WIDTH,3])
          
          image = image * (1. / 127.5) - 1     #(-1,1)
          image = np.reshape(image, [-1,IMAGE_WIDTH*IMAGE_WIDTH*3])
#          image = np.expand_dims(image,0)
          images = np.append(images,image,0)
          m = m+1
          if m>=100:
              break
          
      label = label_to_one_hot(i) 
      labels = np.tile(label,(m,1))
  img_all.append(images)
  lab_all.append(labels)

  current_time = datetime.now().strftime('%Y%m%d-%H%M')
  checkpoints_dir = 'checkpoints/{}'.format(current_time)
  try:
     os.makedirs(checkpoints_dir)
  except os.error:
     pass
 
  with tf.Graph().as_default():
    images = tf.placeholder(tf.float32, [None, IMAGE_WIDTH*IMAGE_WIDTH*3], name='inputs')
    labels = tf.placeholder(tf.float32, [None, class_num], name='labels')
    
    logits = captcha.inference(images, keep_prob=0.75,is_training=True)
    loss = captcha.loss(logits, labels)
#    correct = captcha.evaluation(logits, labels)#train

#    train_precision = correct/batch
    tf.summary.scalar('loss', loss)
#    tf.summary.scalar('train_precision', train_precision)
#    tf.summary.image('images',images,10)
    summary = tf.summary.merge_all()
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()
    
    with tf.Session() as sess:
#            saver.restore(sess, tf.train.latest_checkpoint('ckpt_fm2'))
            sess.run(init)
            summary_writer = tf.summary.FileWriter(train_dir, sess.graph)
            try:
                shuffle_test= inputs(epoch,batch,img_all,lab_all)
                for step, (x_batch, l_batch) in enumerate(shuffle_test):
                    
                    feed_dict = {images:x_batch, labels:l_batch}
                    _, loss_ = sess.run([train_op, loss], feed_dict=feed_dict)
    
                    summary_str = sess.run(summary, feed_dict=feed_dict)
                    summary_writer.add_summary(summary_str, step)
                    summary_writer.flush()
                    
                    if step % 100 == 0:
                        print('>> Step %d run_test: batch_precision = %.2f '
                                          % (step,step))
                    if step % 500 == 0 :
                            saver.save(sess, checkpoint_file, global_step=step)
            except KeyboardInterrupt:
                print('INTERRUPTED')

            finally:
                saver.save(sess, checkpoint_file, global_step=step)
                print('Model saved in file :%s'%FLAGS.checkpoint)

def main(_):
#  if tf.gfile.Exists(FLAGS.train_dir):
#    tf.gfile.DeleteRecursively(FLAGS.train_dir)
#  tf.gfile.MakeDirs(FLAGS.train_dir)
    run_train()


if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--batch_size',
      type=int,
      default=8,
      help='Batch size.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='pig_train',
      help='Directory where to write event logs.'
  )
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='checkpoint/model.ckpt',
      help='Directory where to write checkpoint.'
  )

  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
