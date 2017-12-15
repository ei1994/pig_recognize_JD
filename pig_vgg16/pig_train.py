from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import time
from datetime import datetime
import argparse
import sys
import tensorflow as tf
import pig_model_dark as captcha
#import pig_model_vgg16 as captcha
#import pig_model_restnet_v2 as captcha
import logging

learning_rate = 2e-4
FLAGS = None
def initLogging(logFilename='record.log'):
  """Init for logging
  """
  logging.basicConfig(
                    level    = logging.DEBUG,
                    format='%(asctime)s-%(levelname)s-%(message)s',
                    datefmt  = '%y-%m-%d %H:%M',
                    filename = logFilename,
                    filemode = 'w');
  console = logging.StreamHandler()
  console.setLevel(logging.INFO)
  formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
  console.setFormatter(formatter)
  logging.getLogger('').addHandler(console)
initLogging()

def run_train():
  """Train CAPTCHA for a number of steps."""

  with tf.Graph().as_default():
    images, labels = captcha.inputs(train=True, batch_size=FLAGS.batch_size)

    logits = captcha.inference(images, keep_prob=0.9,is_training=True)
    loss = captcha.loss(logits, labels)
    correct = captcha.evaluation(logits, labels)#train
    tf.summary.scalar('loss', loss)
    summary = tf.summary.merge_all()

#    train_precision = correct/FLAGS.batch_size
#    train_op = captcha.training(loss)
    train_op = tf.train.AdamOptimizer(learning_rate).minimize(loss)
    
    saver = tf.train.Saver()
    init = tf.global_variables_initializer()

    sess = tf.Session()
    sess.run(init)
    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)
#    saver.restore(sess, tf.train.latest_checkpoint(FLAGS.checkpoint_dir))
    coord = tf.train.Coordinator()
    threads = tf.train.start_queue_runners(sess=sess, coord=coord)
    try:
      step = 0
      while not coord.should_stop():
        start_time = time.time()
        _, loss_value, pre_value, logits_ ,labels_= sess.run([train_op, loss, correct,logits,labels])
                    
        duration = time.time() - start_time
        step += 1
        if step % 10 == 0:
          logging.info('>> Step %d run_train: loss = %.2f, train = %.2f (%.3f sec)'
                % (step, loss_value, pre_value, duration))
          summary_str = sess.run(summary)
          summary_writer.add_summary(summary_str, step)
          summary_writer.flush()
          #-------------------------------

        if step % 500 == 0:
          logging.info('>> %s Saving in %s' % (datetime.now(), FLAGS.checkpoint))
          saver.save(sess, FLAGS.checkpoint, global_step=step)
          print(images.shape.as_list(),labels.shape.as_list())

        if step>200000:
          break
    except KeyboardInterrupt:
      print('INTERRUPTED')
      coord.request_stop()
    except Exception as e:

      coord.request_stop(e)
    finally:
      saver.save(sess, FLAGS.checkpoint, global_step=step)
      print('Model saved in file :%s'%FLAGS.checkpoint)

      coord.request_stop()
      coord.join(threads)
    sess.close()



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
      default=32,
      help='Batch size.'
  )
  parser.add_argument(
      '--train_dir',
      type=str,
      default='pig_train',
      help='Directory where to write event logs.'
  )
  parser.add_argument(
      '--checkpoint_dir',
      type=str,
      default='./checkpoint',
      help='Directory where to restore checkpoint.'
  )
  parser.add_argument(
      '--checkpoint',
      type=str,
      default='checkpoint/model.ckpt',
      help='Directory where to write checkpoint.'
  )
  FLAGS, unparsed = parser.parse_known_args()
  tf.app.run(main=main, argv=[sys.argv[0]] + unparsed)
