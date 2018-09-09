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
r"""Infers detections on a TFRecord of TFExamples given an inference graph.

Example usage:
  ./infer_detections \
    --input_tfrecord_paths=/path/to/input/tfrecord1,/path/to/input/tfrecord2 \
    --output_tfrecord_path_prefix=/path/to/output/detections.tfrecord \
    --inference_graph=/path/to/frozen_weights_inference_graph.pb

The output is a TFRecord of TFExamples. Each TFExample from the input is first
augmented with detections from the inference graph and then copied to the
output.

The input and output nodes of the inference graph are expected to have the same
types, shapes, and semantics, as the input and output nodes of graphs produced
by export_inference_graph.py, when run with --input_type=image_tensor.

The script can also discard the image pixels in the output. This greatly
reduces the output size and can potentially accelerate reading data in
subsequent processing steps that don't require the images (e.g. computing
metrics).
"""

import os
import itertools
import threading
import Queue
import time
import random

import tensorflow as tf
from object_detection.inference import detection_inference
from object_detection.inference import mask_inference

tf.flags.DEFINE_string('input_tfrecord_paths', None,
                       'A comma separated list of paths to input TFRecords.')
tf.flags.DEFINE_string('output_tfrecord_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_integer('num_file_per_tfrecord', None,
                        'If not set, output all to the same tfrecord file.')

tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
tf.flags.DEFINE_boolean('discard_image_pixels', False,
                        'Discards the images in the output TFExamples. This'
                        ' significantly reduces the output size and is useful'
                        ' if the subsequent tools don\'t need access to the'
                        ' images (e.g. when computing evaluation measures).')
tf.flags.DEFINE_boolean('detect_masks', None,
                        'If true, output inferred masks.')
tf.flags.DEFINE_integer('override_num_detections', None,
                        'If set, this overrides the number of detections written in the graph.')
tf.flags.DEFINE_integer('num_threads', 1,
                        'Number of threads to be used.')


FLAGS = tf.flags.FLAGS

get_writer_lock = threading.Lock()

def get_writer(shard):
  with get_writer_lock:
    output_file = '%s-%.5d' % (FLAGS.output_tfrecord_path, shard)
    while os.path.exists(output_file):
      shard += 1
      output_file = '%s-%.5d' % (FLAGS.output_tfrecord_path, shard)
    writer = tf.python_io.TFRecordWriter(output_file)
    assert os.path.exists(output_file)
  return writer, shard


def write_examples(thread_index, tf_examples_queue, stop_event):
  shard = 0
  num_examples = 0
  tf_record_writer, shard = get_writer(shard)
  try:
    while not (tf_examples_queue.empty() and stop_event.is_set()):
      if tf_examples_queue.empty():
        time.sleep(random.random())
        continue
      try:
        tf.logging.log_every_n(tf.logging.INFO, 'Approximate q size: %d', 100, tf_examples_queue.qsize())
        tf_example = tf_examples_queue.get(timeout=5)
      except Queue.Empty:
        continue
      if isinstance(tf_example, list) or isinstance(tf_example, tuple):
        for current_example in tf_example:
          tf_record_writer.write(current_example.SerializeToString())
          num_examples += 1
      else:
        tf_record_writer.write(tf_example.SerializeToString())
        num_examples += 1
      if FLAGS.num_file_per_tfrecord and num_examples % FLAGS.num_file_per_tfrecord == 0:
        tf_record_writer.close()
        shard += 1
        tf_record_writer, shard = get_writer(shard)
        tf.logging.log_every_n(tf.logging.INFO, 'Thread %d: Wrote %d examples and is currently on shard %d.', 1,
                               thread_index, num_examples, shard)
  finally:
    tf_record_writer.close()




def main(_):

  tf.logging.set_verbosity(tf.logging.INFO)
  inference_class = detection_inference
  if not os.path.exists(os.path.dirname(FLAGS.output_tfrecord_path)):
    tf.gfile.MakeDirs(os.path.dirname(FLAGS.output_tfrecord_path))
  if FLAGS.detect_masks:
    inference_class = mask_inference
    save_image_path = os.path.join(os.path.dirname(FLAGS.output_tfrecord_path), 'sample_images')
    if not os.path.exists(save_image_path):
      tf.gfile.MakeDirs(save_image_path)
  else:
    save_image_path = None



  required_flags = ['input_tfrecord_paths', 'output_tfrecord_path',
                    'inference_graph']
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, ))


  # input_tfrecord_paths = [
  #     v for v in FLAGS.input_tfrecord_paths.split(',') if v]
  input_tfrecord_paths = []
  for v in FLAGS.input_tfrecord_paths.split(','):
    if v:
      input_tfrecord_paths += tf.gfile.Glob(v)
  tf.logging.info('Reading input from %d files', len(input_tfrecord_paths))
  serialized_example_tensor, image_tensor = inference_class.build_input(
      input_tfrecord_paths)
  if FLAGS.detect_masks:
    image_ph, image_ph_encoded = inference_class.build_image_ph()
  else:
    image_ph, image_ph_encoded = None, None
  tf.logging.info('Reading graph and building model...')
  detected_tensors = inference_class.build_inference_graph(
       image_tensor, FLAGS.inference_graph, override_num_detections=FLAGS.override_num_detections)

  tf.logging.info('Running inference and writing output to {}'.format(
      FLAGS.output_tfrecord_path))
  sess.run(tf.local_variables_initializer())
  tf.train.start_queue_runners(sess=sess)
  # MODIFIED:
  # with tf.python_io.TFRecordWriter(
  #     FLAGS.output_tfrecord_path) as tf_record_writer:
  #   try:
  #     for counter in itertools.count():
  #       tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
  #                              counter)
  #       if FLAGS.detect_masks:
  #         tf_example = inference_class.infer_detections_and_add_to_example(
  #           serialized_example_tensor, image_tensor, detected_tensors[0],
  #           detected_tensors[1], detected_tensors[2], detected_tensors[3],
  #           FLAGS.discard_image_pixels, image_ph, image_ph_encoded,
  #           do_save_image=True, save_image_path=save_image_path)
  #       else:
  #         tf_example = inference_class.infer_detections_and_add_to_example(
  #             serialized_example_tensor, detected_tensors[0],
  #             detected_tensors[1], detected_tensors[2],
  #             FLAGS.discard_image_pixels)
  #
  #       if isinstance(tf_example, list) or isinstance(tf_example, tuple):
  #         for current_example in tf_example:
  #           tf_record_writer.write(current_example.SerializeToString())
  #       else:
  #         tf_record_writer.write(tf_example.SerializeToString())
  #   except tf.errors.OutOfRangeError:
  #     tf.logging.info('Finished processing records')

  # def infer(thread_index, sess, inference_class, serialized_example_tensor, image_tensor, detected_tensors, image_ph,
  #           image_ph_encoded, save_image_path):
  #   shard = 0
  #   num_examples = 0
  #   tf_record_writer, shard = get_writer(shard)
  #   # TODO: implement multithreading. Currently multithreading runs out of memory... which I do not understand. Is it trying to
  #   # allocate two copies? In any case the gpu util is still around 60% for 1 thread.Maybe I can just do multithread for output?
  #   try:
  #     for counter in itertools.count():
  #       tf.logging.log_every_n(tf.logging.INFO, 'Thread %d: Processed %d images...', 10,
  #                              thread_index, counter)
  #       if FLAGS.detect_masks:
  #         tf_example = inference_class.infer_detections_and_add_to_example(
  #           sess,
  #           serialized_example_tensor, image_tensor, detected_tensors,
  #           FLAGS.discard_image_pixels, image_ph, image_ph_encoded,
  #           do_save_image=False, save_image_path=save_image_path)
  #       else:
  #         tf_example = inference_class.infer_detections_and_add_to_example(
  #           serialized_example_tensor, detected_tensors[0],
  #           detected_tensors[1], detected_tensors[2],
  #           FLAGS.discard_image_pixels)
  #
  #       if isinstance(tf_example, list) or isinstance(tf_example, tuple):
  #         for current_example in tf_example:
  #           tf_record_writer.write(current_example.SerializeToString())
  #           num_examples += 1
  #       else:
  #         tf_record_writer.write(tf_example.SerializeToString())
  #         num_examples += 1
  #       if FLAGS.num_file_per_tfrecord and num_examples % FLAGS.num_file_per_tfrecord == 0:
  #         tf_record_writer.close()
  #         shard += 1
  #         tf_record_writer, shard = get_writer(shard)
  #   except tf.errors.OutOfRangeError:
  #     tf.logging.info('Finished processing records')
  #   finally:
  #     tf_record_writer.close()

  threads = []
  tf_examples_queue = Queue.Queue()
  stop_event = threading.Event()
  # Create a mechanism for monitoring when all threads are finished.
  coord = tf.train.Coordinator()
  for thread_index in xrange(FLAGS.num_threads):
    args = (thread_index, tf_examples_queue, stop_event)
    t = threading.Thread(target=write_examples, args=args)
    t.start()
    threads.append(t)

  try:
    for counter in itertools.count():
      tf.logging.log_every_n(tf.logging.INFO, 'Processed %d images...', 10,
                             counter)
      if FLAGS.detect_masks:
        tf_example = inference_class.infer_detections_and_add_to_example(
          sess,
          serialized_example_tensor, image_tensor, detected_tensors,
          FLAGS.discard_image_pixels, image_ph, image_ph_encoded,
          do_save_image=False, save_image_path=save_image_path)
      else:
        tf_example = inference_class.infer_detections_and_add_to_example(
          sess, serialized_example_tensor, image_tensor, detected_tensors[0],
          detected_tensors[1], detected_tensors[2],
          FLAGS.discard_image_pixels)
      tf_examples_queue.put(tf_example)
  except tf.errors.OutOfRangeError:
    tf.logging.info('Finished processing records')

  # Wait for all the threads to terminate.
  stop_event.set()
  coord.join(threads)
  print('Finished processing all images in data set.')
  # MODIFIED ENDS


if __name__ == '__main__':
  tf.app.run()

# GPU 0 110 files => 7 hours.
# GPU 1 110 files => 9 hours.