# Copyright 2017 The TensorFlow Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
"""Utility functions for detection inference."""
from __future__ import division

import time

import tensorflow as tf
import numpy as np
import scipy.misc
import os

from object_detection.core import standard_fields
from object_detection.core import prefetcher
from object_detection.utils import visualization_utils as vis_utils
from object_detection.utils import ops
from object_detection.builders import image_resizer_builder
from object_detection.protos import image_resizer_pb2
from google.protobuf import text_format

DEFAULT_CATEGORY_INDEX = {1: {'id': 1, 'name': 'anime_figure'}}

def build_input(tfrecord_paths):
  """Builds the graph's input.

  Args:
    tfrecord_paths: List of paths to the input TFRecords

  Returns:
    serialized_example_tensor: The next serialized example. String scalar Tensor
    image_tensor: The decoded image of the example. Uint8 tensor,
        shape=[1, None, None,3]
  """
  filename_queue = tf.train.string_input_producer(
      tfrecord_paths, shuffle=False, num_epochs=1)

  tf_record_reader = tf.TFRecordReader()
  _, serialized_example_tensor = tf_record_reader.read(filename_queue)

  # *** MODIFIED
  prefetch_queue = prefetcher.prefetch({'serialized_example_tensor': serialized_example_tensor}, 100)
  dequeue = prefetch_queue.dequeue()
  serialized_example_tensor = dequeue['serialized_example_tensor']

  # *** MODIFIED ENDS



  features = tf.parse_single_example(
      serialized_example_tensor,
      features={
          standard_fields.TfExampleFields.image_encoded:
              tf.FixedLenFeature([], tf.string),
      })
  encoded_image = features[standard_fields.TfExampleFields.image_encoded]
  image_tensor = tf.image.decode_image(encoded_image, channels=3)
  image_tensor.set_shape([None, None, 3])
  # image_tensor = tf.expand_dims(image_tensor, 0)

  # # *** MODIFIED
  # batch = tf.train.batch(
  #   [serialized_example_tensor, image_tensor],
  #   batch_size=24,
  #   enqueue_many=False,
  #   num_threads=6,
  #   capacity=5 * 24)
  # return batch[0], batch[1]
  image_resizer_text_proto = """
    keep_aspect_ratio_resizer {
      min_dimension: 800
      max_dimension: 1365
    }
  """
  image_resizer_config = image_resizer_pb2.ImageResizer()
  text_format.Merge(image_resizer_text_proto, image_resizer_config)
  image_resizer_fn = image_resizer_builder.build(image_resizer_config)
  resized_image_tensor, _ = image_resizer_fn(image_tensor)
  # resized_image_tensor = tf.image.convert_image_dtype(resized_image_tensor, dtype=tf.uint8)
  resized_image_tensor = tf.cast(resized_image_tensor, dtype=tf.uint8)
  resized_image_tensor = tf.expand_dims(resized_image_tensor, 0)

  # # *** MODIFIED ENDS


  return serialized_example_tensor, resized_image_tensor#image_tensor

def build_image_ph():
  image_ph = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3], name='image_ph')
  image_ph_encoded = tf.image.encode_png(image_ph)
  return image_ph, image_ph_encoded


def build_inference_graph(image_tensor, inference_graph_path, override_num_detections=None):
  """Loads the inference graph and connects it to the input image.

  Args:
    image_tensor: The input image. uint8 tensor, shape=[1, None, None, 3]
    inference_graph_path: Path to the inference graph with embedded weights

  Returns:
    detected_boxes_tensor: Detected boxes. Float tensor,
        shape=[num_detections, 4]
    detected_scores_tensor: Detected scores. Float tensor,
        shape=[num_detections]
    detected_labels_tensor: Detected labels. Int64 tensor,
        shape=[num_detections]
  """
  with tf.gfile.Open(inference_graph_path, 'rb') as graph_def_file:
    graph_content = graph_def_file.read()
  graph_def = tf.GraphDef()
  graph_def.MergeFromString(graph_content)

  tf.import_graph_def(
      graph_def, name='', input_map={'image_tensor': image_tensor})

  g = tf.get_default_graph()

  if override_num_detections is not None:
    num_detections_tensor = tf.cast(override_num_detections, tf.int32)
  else:
    num_detections_tensor = tf.squeeze(
        g.get_tensor_by_name('num_detections:0'), 0)
    num_detections_tensor = tf.cast(num_detections_tensor, tf.int32)

  detected_boxes_tensor = tf.squeeze(
      g.get_tensor_by_name('detection_boxes:0'), 0)
  detected_boxes_tensor = detected_boxes_tensor[:num_detections_tensor]

  detected_scores_tensor = tf.squeeze(
      g.get_tensor_by_name('detection_scores:0'), 0)
  detected_scores_tensor = detected_scores_tensor[:num_detections_tensor]

  detected_labels_tensor = tf.squeeze(
      g.get_tensor_by_name('detection_classes:0'), 0)
  detected_labels_tensor = tf.cast(detected_labels_tensor, tf.int64)
  detected_labels_tensor = detected_labels_tensor[:num_detections_tensor]

  detected_masks_tensor = tf.squeeze(
      g.get_tensor_by_name('detection_masks:0'), 0)
  # detected_masks_tensor = tf.cast(detected_masks_tensor, tf.int32)

  image_shape = tf.shape(image_tensor)
  detected_masks_tensor = tf.slice(
    detected_masks_tensor, begin=[0, 0, 0], size=[num_detections_tensor, -1, -1])
  detection_masks_reframed = ops.reframe_box_masks_to_image_masks(
    detected_masks_tensor, detected_boxes_tensor, image_shape[1], image_shape[2])
  detection_masks_reframed = tf.cast(
    tf.greater(detection_masks_reframed, 0.5), tf.uint8)

  # For some unknown reason directly feeding in detection_masks_reframed makes tensorflow stuck...
  ph = tf.placeholder(dtype=tf.uint8, shape=[None, None, None])
  detection_masks_encoded = tf.map_fn(lambda x: tf.image.encode_png(x), tf.expand_dims(ph, axis=-1), dtype=tf.string)



  return detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor, detection_masks_reframed, detection_masks_encoded, ph


def infer_detections_and_add_to_example(
    sess, serialized_example_tensor, image_tensor, detected_tensors, discard_image_pixels, image_ph, image_ph_encoded,
    do_save_image=False, save_image_path=''):
  """Runs the supplied tensors and adds the inferred detections to the example.

  Args:
    serialized_example_tensor: Serialized TF example. Scalar string tensor
    detected_boxes_tensor: Detected boxes. Float tensor,
        shape=[num_detections, 4]
    detected_scores_tensor: Detected scores. Float tensor,
        shape=[num_detections]
    detected_labels_tensor: Detected labels. Int64 tensor,
        shape=[num_detections]
    discard_image_pixels: If true, discards the image from the result
  Returns:
    The de-serialized TF example augmented with the inferred detections.
  """
  (detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor, detected_masks_tensor,
   detection_masks_encoded, ph) = detected_tensors
  tf_example = tf.train.Example()
  run_list = [
       serialized_example_tensor, image_tensor, detected_boxes_tensor, detected_scores_tensor,
       detected_labels_tensor, detected_masks_tensor,
   ]

  output_list = sess.run(run_list)
  (serialized_example, image, detected_boxes, detected_scores,
   detected_classes, detected_masks) = output_list[:6]
  masks_encoded, = sess.run([detection_masks_encoded], feed_dict={ph: detected_masks})

  detected_boxes = detected_boxes.T

  tf_example.ParseFromString(serialized_example)
  feature = tf_example.features.feature
  feature[standard_fields.TfExampleFields.
          detection_score].float_list.value[:] = detected_scores
  feature[standard_fields.TfExampleFields.
          detection_bbox_ymin].float_list.value[:] = detected_boxes[0]
  feature[standard_fields.TfExampleFields.
          detection_bbox_xmin].float_list.value[:] = detected_boxes[1]
  feature[standard_fields.TfExampleFields.
          detection_bbox_ymax].float_list.value[:] = detected_boxes[2]
  feature[standard_fields.TfExampleFields.
          detection_bbox_xmax].float_list.value[:] = detected_boxes[3]
  feature[standard_fields.TfExampleFields.
          detection_class_label].int64_list.value[:] = detected_classes
  feature[standard_fields.TfExampleFields.
          instance_masks].bytes_list.value[:] = masks_encoded

  if do_save_image:
    # TODO: hard coded for our purpose for now.
    category_index = {1: {'id': 1, 'name': 'anime_figure'}}
    annotated_image = vis_utils.visualize_boxes_and_labels_on_image_array(
        image[0],
      detected_boxes.T,
      detected_classes,
      detected_scores,
        category_index,
        instance_masks=detected_masks,
        use_normalized_coordinates=True,
      min_score_thresh=.5,
      max_boxes_to_draw=20,
        agnostic_mode=False,
        skip_scores=True,
        skip_labels=True)
    # For debugging.
    scipy.misc.imsave(
      os.path.join(save_image_path, 'mask_sample_%d.png' %int(time.time())),
      annotated_image)
    encoded_annotated_image = sess.run([image_ph_encoded], feed_dict={image_ph: annotated_image})
    feature['image/encoded_annotated_image'].bytes_list.value[:] = encoded_annotated_image


  if discard_image_pixels:
    del feature[standard_fields.TfExampleFields.image_encoded]

  return tf_example


  # # *** MODIFIED
  # This does not work because each image inthe batch has a different size!
  # (serialized_example_batched, detected_boxes_batched, detected_scores_batched,
  #  detected_classes_batched) = sess.run([
  #      serialized_example_tensor, detected_boxes_tensor, detected_scores_tensor,
  #      detected_labels_tensor
  #  ])
  # ret = []
  # for (serialized_example, detected_boxes, detected_scores,
  #  detected_classes) in zip(serialized_example_batched, detected_boxes_batched, detected_scores_batched,
  #  detected_classes_batched):
  #   tf_example = tf.train.Example()
  #   detected_boxes = detected_boxes.T
  #
  #   tf_example.ParseFromString(serialized_example)
  #   feature = tf_example.features.feature
  #   feature[standard_fields.TfExampleFields.
  #           detection_score].float_list.value[:] = detected_scores
  #   feature[standard_fields.TfExampleFields.
  #           detection_bbox_ymin].float_list.value[:] = detected_boxes[0]
  #   feature[standard_fields.TfExampleFields.
  #           detection_bbox_xmin].float_list.value[:] = detected_boxes[1]
  #   feature[standard_fields.TfExampleFields.
  #           detection_bbox_ymax].float_list.value[:] = detected_boxes[2]
  #   feature[standard_fields.TfExampleFields.
  #           detection_bbox_xmax].float_list.value[:] = detected_boxes[3]
  #   feature[standard_fields.TfExampleFields.
  #           detection_class_label].int64_list.value[:] = detected_classes
  #
  #   if discard_image_pixels:
  #     del feature[standard_fields.TfExampleFields.image_encoded]
  #   ret.append(tf_example)
  #
  # return ret

def infer_detections(
    sess, image_tensor, detected_tensors,
    min_score_thresh=.5, visualize_inference=False, category_index=None, feed_dict=None):
  """Runs the supplied tensors and adds the inferred detections to the example.

  Args:
    serialized_example_tensor: Serialized TF example. Scalar string tensor
    detected_boxes_tensor: Detected boxes. Float tensor,
        shape=[num_detections, 4]
    detected_scores_tensor: Detected scores. Float tensor,
        shape=[num_detections]
    detected_labels_tensor: Detected labels. Int64 tensor,
        shape=[num_detections]
    discard_image_pixels: If true, discards the image from the result
  Returns:
    The de-serialized TF example augmented with the inferred detections.
  """
  (detected_boxes_tensor, detected_scores_tensor, detected_labels_tensor, detected_masks_tensor,
   detection_masks_encoded, ph) = detected_tensors
  run_list = [
       image_tensor, detected_boxes_tensor, detected_scores_tensor,
       detected_labels_tensor, detected_masks_tensor,
   ]

  output_list = sess.run(run_list, feed_dict=feed_dict)
  (image, detected_boxes, detected_scores, detected_classes, detected_masks) = output_list[:6]
  detected_boxes = detected_boxes.T


  indices = detected_scores > min_score_thresh

  ret = {
    'detection_score': detected_scores[indices].tolist(),
    'detection_bbox_ymin': detected_boxes[0][indices].tolist(),
    'detection_bbox_xmin': detected_boxes[1][indices].tolist(),
    'detection_bbox_ymax': detected_boxes[2][indices].tolist(),
    'detection_bbox_xmax': detected_boxes[3][indices].tolist(),
    'detection_class_label': detected_classes[indices].tolist(),
    'detected_masks': detected_masks[indices].tolist()
  }

  if visualize_inference:
    annotated_image = vis_utils.visualize_boxes_and_labels_on_image_array(
      image[0],
      detected_boxes.T,
      detected_classes,
      detected_scores,
      category_index or DEFAULT_CATEGORY_INDEX,
      instance_masks=detected_masks,
      use_normalized_coordinates=True,
      min_score_thresh=min_score_thresh,
      max_boxes_to_draw=20,
      agnostic_mode=False,
      skip_scores=False,
      skip_labels=False)

    ret['annotated_image'] = annotated_image
  return ret