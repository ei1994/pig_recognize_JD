# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 12:27:34 2017

@author: DELL
"""
'''
使用 darknet19网络
'''
#from __future__ import absolute_import
#from __future__ import division
#from __future__ import print_function

import tensorflow as tf
import pig_input
import config

IMAGE_WIDTH = config.IMAGE_WIDTH
IMAGE_HEIGHT = config.IMAGE_HEIGHT
CLASSES_NUM = config.CLASSES_NUM
rate = 0.01

def inputs(train, batch_size):
    return pig_input.inputs(train, batch_size=batch_size)

def _conv(name, input, size, input_channels, output_channels, is_training=True):
  with tf.variable_scope(name) as scope:
    if not is_training:
      scope.reuse_variables()
    kernel = _weight_variable('weights', shape=[size, size ,input_channels, output_channels])
    biases = _bias_variable('biases',[output_channels])
    pre_activation = tf.nn.bias_add(_conv2d(input, kernel),biases)
    conv = tf.nn.relu(pre_activation)
#    conv = tf.maximum(rate*pre_activation,pre_activation, name=scope.name)
#    conv = _batch_norm('norm', conv, is_training)
  return conv

def _conv2d(value, weight):
  """conv2d returns a 2d convolution layer with full stride."""
  return tf.nn.conv2d(value, weight, strides=[1, 1, 1, 1], padding='SAME')


def _max_pool_2x2(value, name, is_training):
  """max_pool_2x2 downsamples a feature map by 2X."""
  with tf.variable_scope(name) as scope1:
    if not is_training:
      scope1.reuse_variables()
    return tf.nn.max_pool(value, ksize=[1, 2, 2, 1],
                        strides=[1, 2, 2, 1], padding='SAME', name=name)

def _weight_variable(name, shape):
  """weight_variable generates a weight variable of a given shape."""
  initializer = tf.truncated_normal_initializer(stddev=0.1)
  var = tf.get_variable(name,shape,initializer=initializer, dtype=tf.float32)
  return var


def _bias_variable(name, shape):
  """bias_variable generates a bias variable of a given shape."""
  initializer = tf.constant_initializer(0.1)
  var = tf.get_variable(name, shape, initializer=initializer,dtype=tf.float32)
  return var

def _batch_norm(name, inputs, is_training):
  """ Batch Normalization
  """
  with tf.variable_scope(name, reuse = not is_training):
#      return tf.layers.batch_normalization(input,training=is_training)
    return tf.contrib.layers.batch_norm(inputs,
                                        decay=0.9,
                                        scale=True,
                                        updates_collections=None,
                                        is_training=True)
def inference(images, keep_prob, is_training):
    
      images = tf.reshape(images, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])  # 320,320,3
      
      conv1 = _conv('conv1', images, 3, 3, 32, is_training)
#      conv1 = _batch_norm('norm', conv1, is_training)  
      pool1 = _max_pool_2x2(conv1, name='pool1', is_training=is_training)
      
      conv2 = _conv('conv2', pool1, 3, 32, 64, is_training)
      conv2 = _batch_norm('norm1', conv2, is_training)  
      pool2 = _max_pool_2x2(conv2, name='pool2', is_training=is_training)
      
      conv3 = _conv('conv3', pool2, 3, 64, 128, is_training)  # 80*80*128
      conv4 = _conv('conv4', conv3, 1, 128, 64, is_training)
      conv5 = _conv('conv5', conv4, 3, 64, 128, is_training)
      conv5 = _batch_norm('norm2', conv5, is_training)   
      
      pool3 = _max_pool_2x2(conv5, name='pool3', is_training=is_training)
      
      conv6 = _conv('conv6', pool3, 3, 128, 256, is_training)  # 40*40*256
      conv7 = _conv('conv7', conv6, 1, 256, 128, is_training)
      conv8 = _conv('conv8', conv7, 3, 128, 256, is_training)
      conv8 = _batch_norm('norm3', conv8, is_training) 
        
      pool4 = _max_pool_2x2(conv8, name='pool4', is_training=is_training) # 8,28
      
      conv9 = _conv('conv9', pool4, 3, 256, 512, is_training)   # 20*20*512
      conv10 = _conv('conv10', conv9, 1, 512, 256, is_training)
      conv11 = _conv('conv11', conv10, 3, 256, 512, is_training)
      conv12 = _conv('conv12', conv11, 1, 512, 256, is_training)
      conv13 = _conv('conv13', conv12, 3, 256, 512, is_training)
      conv13 = _batch_norm('norm4', conv13, is_training) 

      pool5 = _max_pool_2x2(conv13, name='pool5', is_training=is_training) # 8,28
      
      conv14 = _conv('conv14', pool5, 3, 512, 1024, is_training)    # 10*10*1024
      conv15 = _conv('conv15', conv14, 1, 1024, 512, is_training)
      conv16 = _conv('conv16', conv15, 3, 512, 1024, is_training)
      conv17 = _conv('conv17', conv16, 1, 1024, 512, is_training)
      conv18 = _conv('conv18', conv17, 3, 512, 1024, is_training)
      conv18 = _batch_norm('norm5', conv18, is_training) 
      
      pool6 = _max_pool_2x2(conv18, name='pool6', is_training=is_training) # 5*5*1024
      batch_size = int(pool6.get_shape()[0])
      dense = tf.reshape(pool5, [batch_size,-1])
      dense1 = tf.layers.dense(dense, 1024)
      dense1 = tf.nn.relu(dense1)
      dense1 = tf.nn.dropout(dense1, keep_prob)
      dense2 = tf.layers.dense(dense1, 256)
      dense2 = tf.nn.relu(dense2)
      dense2 = tf.nn.dropout(dense2, keep_prob)
      dense3 = tf.layers.dense(dense2, 30)
#      output = tf.nn.sigmoid(dense3)
#      output = tf.nn.softmax(dense3)
      
      return dense3

    
def loss(logits, labels):
  cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
                  labels=labels, logits=logits, name='corss_entropy_per_example')
  cross_entropy_mean = tf.reduce_mean(cross_entropy, name='cross_entropy')
  tf.add_to_collection('losses', cross_entropy_mean)
  return tf.add_n(tf.get_collection('losses'), name='total_loss')


def training(loss):
  optimizer = tf.train.AdamOptimizer(1e-4)
  gen_grads_and_vars = optimizer.compute_gradients(loss)
  gen_train = optimizer.apply_gradients(gen_grads_and_vars)
  ema = tf.train.ExponentialMovingAverage(decay=0.99)
  update_losses = ema.apply([loss])

  global_step = tf.contrib.framework.get_or_create_global_step()
  incr_global_step = tf.assign(global_step, global_step+1)

  return tf.group(update_losses, incr_global_step, gen_train)



def evaluation(logits, labels):
  correct_prediction = tf.equal(tf.argmax(logits,1), tf.argmax(labels,1))
  return tf.reduce_sum(tf.cast(correct_prediction, tf.float32))


def output(logits):
  return tf.nn.softmax(logits)

def predict(logits):
  return tf.argmax(logits, 1)