# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 15:07:52 2017

@author: DELL
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 30 16:16:13 2017

@author: no1
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

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
    conv = tf.maximum(rate*pre_activation,pre_activation, name=scope.name)
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
  images = tf.reshape(images, [-1, IMAGE_HEIGHT, IMAGE_WIDTH, 3])  # 256,256,3
#%% vgg16
    # conv1_1

  conv1_1 = _conv('conv1_1', images, 3, 3, 64,is_training) #(batch,256,256,64)
  conv1_2 = _conv('conv1_2', conv1_1,3, 64,64,is_training)#(batch,256,256,64)
  pool1 = _max_pool_2x2(conv1_2,'pool1',is_training)



  conv2_1 = _conv('conv2_1', pool1, 3,64,128,is_training) #(60, 128, 128, 128)
  conv2_2 = _conv('conv2_2', conv2_1,3,128,128,is_training) #(60, 128, 128, 128)
  pool2 = _max_pool_2x2(conv2_2, 'pool2',is_training)


  conv3_1 = _conv('conv3_1', pool2, 3, 128, 256,is_training) #(60, 64, 64, 256)
  conv3_2 = _conv('conv3_2',conv3_1, 3, 256, 256,is_training)#(60, 64, 64, 256)
  conv3_3 = _conv('conv3_3',conv3_2, 3, 256, 256,is_training)#(60, 64, 64, 256)
  pool3 = _max_pool_2x2(conv3_3, 'pool3',is_training)


  conv4_1 = _conv('conv4_1',pool3, 3, 256, 512,is_training)
  conv4_2 = _conv('conv4_2',conv4_1, 3, 512, 512,is_training)
  conv4_3 = _conv('conv4_3',conv4_2, 3, 512, 512,is_training)
  pool4 = _max_pool_2x2(conv4_3, 'pool4',is_training)

  conv5_1 = _conv('conv5_1',pool4, 3, 512, 512,is_training)
  conv5_2 = _conv('conv5_2',conv5_1, 3, 512, 512,is_training)
  conv5_3 = _conv('conv5_3',conv5_2, 3, 512, 128,is_training)
  pool5 = _max_pool_2x2(conv5_3, 'pool5',is_training)  #(batch,14,14,512)
  norm = _batch_norm('norm', pool5, is_training)
  #%%
  with tf.variable_scope('local1') as scope14:
    if not is_training:
      scope14.reuse_variables()
    tensor_shape = norm.get_shape().as_list()
    reshape = tf.reshape(norm, [-1, tensor_shape[1]*tensor_shape[2]*tensor_shape[3]])
    dim = reshape.get_shape()[1].value
    weights = _weight_variable('weights', shape=[dim,1024])
    biases = _bias_variable('biases',[1024])
    local1 = tf.nn.relu(tf.matmul(reshape,weights) + biases, name=scope14.name)

    local1_drop = tf.nn.dropout(local1, keep_prob)

  with tf.variable_scope('softmax_linear') as scope15:
    if not is_training:
      scope15.reuse_variables()
    weights = _weight_variable('weights',shape=[1024,CLASSES_NUM])
    biases = _bias_variable('biases',[CLASSES_NUM])
    softmax_linear = tf.add(tf.matmul(local1_drop,weights), biases, name=scope15.name)

  return tf.reshape(softmax_linear, [-1, CLASSES_NUM])


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