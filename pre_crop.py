# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 09:38:39 2017

@author: DELL
"""

import tensorflow as tf
import scipy.misc as misc
import matplotlib.pylab as plt

def preprocess_for_eval_beifen(image, height, width,
                        central_fraction=0.875, scope=None):
  """Prepare one image for evaluation.

  If height and width are specified it would output an image with that size by
  applying resize_bilinear.

  If central_fraction is specified it would crop the central fraction of the
  input image.

  Args:
    image: 3-D Tensor of image. If dtype is tf.float32 then the range should be
      [0, 1], otherwise it would converted to tf.float32 assuming that the range
      is [0, MAX], where MAX is largest positive representable number for
      int(8/16/32) data type (see `tf.image.convert_image_dtype` for details).
    height: integer
    width: integer
    central_fraction: Optional Float, fraction of the image to crop.
    scope: Optional scope for name_scope.
  Returns:
    3-D float Tensor of prepared image.
  """
  with tf.name_scope(scope, 'eval_image', [image, height, width]):
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Crop the central region of the image with an area containing 87.5% of
    # the original image.

    if central_fraction:
      image = tf.image.central_crop(image, central_fraction=central_fraction)

#    if height and width:
#      # Resize the image to the specified height and width.
#      image = tf.expand_dims(image, 0)
#      image = tf.image.resize_bilinear(image, [height, width], align_corners = False)
#      image = tf.squeeze(image, [0])
##    image = tf.subtract(image, 0.5)
##    image = tf.multiply(image, 2.0)
    image_1 = tf.image.convert_image_dtype(image, dtype=tf.uint8)
    return image_1

def distorted_bounding_box_crop(image,
                                bbox=None,
                                min_object_covered=0.1,
                                aspect_ratio_range=(0.75, 1.33),
                                area_range=(0.05, 1.0),
                                max_attempts=100,
                                scope=None):
  """Generates cropped_image using a one of the bboxes randomly distorted.

  """
  with tf.name_scope(scope, 'distorted_bounding_box_crop', [image, bbox]):
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].

    # A large fraction of image datasets contain a human-annotated bounding
    # box delineating the region of the image containing the object of interest.
    # We choose to create a new bounding box for the object which is a randomly
    # distorted version of the human-annotated bounding box that obeys an
    # allowed range of aspect ratios, sizes and overlap with the human-annotated
    # bounding box. If no box is supplied, then we assume the bounding box is
    # the entire image.
    sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
        tf.shape(image),
        bounding_boxes=bbox,
        min_object_covered=min_object_covered,
        aspect_ratio_range=aspect_ratio_range,
        area_range=area_range,
        max_attempts=max_attempts,
        use_image_if_no_bounding_boxes=True)
    bbox_begin, bbox_size, distort_bbox = sample_distorted_bounding_box

    # Crop the image to the specified bounding box.
    cropped_image = tf.slice(image, bbox_begin, bbox_size)
    return cropped_image, distort_bbox

def preprocess_for_train(image, height, width, bbox=None,
                         fast_mode=True,
                         scope=None,
                         add_image_summaries=True):    
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0],
                         dtype=tf.float32,
                         shape=[1, 1, 4])
    if image.dtype != tf.float32:
      image = tf.image.convert_image_dtype(image, dtype=tf.float32)
    # Each bounding box has shape [1, num_boxes, box coords] and
    # the coordinates are ordered [ymin, xmin, ymax, xmax].
    image_with_box = tf.image.draw_bounding_boxes(tf.expand_dims(image, 0),
                                                  bbox)
    distorted_image, distorted_bbox = distorted_bounding_box_crop(image, bbox)
    # Restore the shape since the dynamic slice based upon the bbox_size loses
    # the third dimension.
    distorted_image.set_shape([None, None, 3])
    image_with_distorted_box = tf.image.draw_bounding_boxes(
        tf.expand_dims(image, 0), distorted_bbox)
    return image_with_box, image_with_distorted_box

if __name__ == '__main__':
    graph_a = tf.Graph()
    with graph_a.as_default():
        img = plt.imread('D:/pig_recognize/pig_slim1/pig_test/00031.JPG')
        
        img_input = tf.placeholder(dtype=tf.float32 )
#        image = preprocess_for_eval_beifen(img_input, 299, 299)
        image1, image2 = preprocess_for_train(img_input, height=299, width=299)

        init = tf.global_variables_initializer()
        
        with tf.Session() as sess:
            sess.run(init)
            image1_, image2_ = sess.run([image1, image2], feed_dict={img_input:img})
            image_1 = image1_[0]
            image_2 = image2_[0]
        plt.imsave('1.jpg',image_1)
        plt.imsave('2.jpg',image_2)
    
    
    
    
    
    
    