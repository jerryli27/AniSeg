"""Does object detection and segmentation on images."""
import json
import os
import threading

import numpy as np
import tensorflow as tf
from google.protobuf import text_format

import util_io
from object_detection.builders import image_resizer_builder
from object_detection.inference import detection_inference
from object_detection.inference import mask_inference
from object_detection.protos import image_resizer_pb2

tf.flags.DEFINE_string('input_images', None,
                       'A comma separated list of paths/patterns to input images.'
                       'e.g. "PATH/WITH/IMAGES/*,ANOTHER/PATH/1.jpg"')
tf.flags.DEFINE_string('output_path', None,
                       'Path to the output TFRecord.')
tf.flags.DEFINE_boolean('visualize_inference', False,
                        'If set, also outputs the annotated inference result image.')
tf.flags.DEFINE_boolean('output_cropped_image', False,
                        'If set, also outputs the cropped image to the output path. e.g. '
                        'OUTPUT_PATH/IMAGE_NAME_crop.png.')
tf.flags.DEFINE_boolean('only_output_cropped_single_object', False,
                        'Only used if FLAGS.output_cropped_image is True. Only outputs the cropped image if there is '
                        'one and only one object detected.')

tf.flags.DEFINE_string('inference_graph', None,
                       'Path to the inference graph with embedded weights.')
tf.flags.DEFINE_boolean('detect_masks', None,
                        'If true, output inferred masks.')
tf.flags.DEFINE_integer('override_num_detections', None,
                        'If set, this overrides the number of detections written in the graph.')
tf.flags.DEFINE_float('min_score_thresh', 0.5,
                      'Minimum score. Detection proposals below this score are discarded.')

FLAGS = tf.flags.FLAGS

get_writer_lock = threading.Lock()


def build_input():
  image_tensor = image_ph = tf.placeholder(dtype=tf.uint8, shape=[None, None, 3], name='image_ph')
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
  resized_image_tensor = tf.cast(resized_image_tensor, dtype=tf.uint8)
  resized_image_tensor = tf.expand_dims(resized_image_tensor, 0)

  return image_ph, resized_image_tensor


def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)
  inference_class = mask_inference if FLAGS.detect_masks else detection_inference
  if not os.path.exists(FLAGS.output_path):
    tf.gfile.MakeDirs(FLAGS.output_path)

  required_flags = ['input_images', 'output_path',
                    'inference_graph']
  for flag_name in required_flags:
    if not getattr(FLAGS, flag_name):
      raise ValueError('Flag --{} is required'.format(flag_name))

  sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True, ))

  input_image_paths = []
  for v in FLAGS.input_images.split(','):
    if v:
      input_image_paths += tf.gfile.Glob(v)
  tf.logging.info('Reading input from %d files', len(input_image_paths))
  image_ph, image_tensor = build_input()

  tf.logging.info('Reading graph and building model...')
  detected_tensors = inference_class.build_inference_graph(
    image_tensor, FLAGS.inference_graph, override_num_detections=FLAGS.override_num_detections)

  tf.logging.info('Running inference and writing output to {}'.format(
    FLAGS.output_path))
  sess.run(tf.local_variables_initializer())

  for i, image_path in enumerate(input_image_paths):
    image_np = util_io.imread(image_path)
    result = inference_class.infer_detections(
      sess, image_tensor, detected_tensors,
      min_score_thresh=FLAGS.min_score_thresh,
      visualize_inference=FLAGS.visualize_inference,
      feed_dict={image_ph: image_np})

    if FLAGS.output_cropped_image:
      if FLAGS.only_output_cropped_single_object and len(result["detection_score"]) == 1:
        num_outputs = 1
      else:
        num_outputs = len(result["detection_score"])

      for crop_i in range(0, num_outputs):
        if (result["detection_score"])[crop_i] > FLAGS.min_score_thresh:
          base, ext = os.path.splitext(os.path.basename(image_path))
          output_crop = os.path.join(FLAGS.output_path, base + '_crop_%d.png' %crop_i)
          idims = image_np.shape  # np array with shape (height, width, num_color(1, 3, or 4))
          min_x = int(min(round(result["detection_bbox_xmin"][crop_i] * idims[1]), idims[1]))
          max_x = int(min(round(result["detection_bbox_xmax"][crop_i] * idims[1]), idims[1]))
          min_y = int(min(round(result["detection_bbox_ymin"][crop_i] * idims[0]), idims[0]))
          max_y = int(min(round(result["detection_bbox_ymax"][crop_i] * idims[0]), idims[0]))
          image_cropped = image_np[min_y:max_y, min_x:max_x, :]
          util_io.imsave(output_crop, image_cropped)

    if FLAGS.visualize_inference:
      output_image = os.path.join(FLAGS.output_path, os.path.basename(image_path))
      util_io.imsave(output_image, result['annotated_image'])
      del result['annotated_image']  # No need to write the image to json.
    if FLAGS.detect_masks:
      base, ext = os.path.splitext(os.path.basename(image_path))
      for mask_i in range(len(result['detected_masks'])):
        # Stores as png to preserve accurate mask values.
        output_mask = os.path.join(FLAGS.output_path, base + '_mask_%d' % mask_i + '.png')
        util_io.imsave(output_mask, np.array(result['detected_masks'][mask_i]) * 255)
      del result['detected_masks']  # Storing mask in json is pretty space consuming.

    output_file = os.path.join(FLAGS.output_path, os.path.splitext(os.path.basename(image_path))[0] + '.json')
    with open(output_file, 'w') as f:
      json.dump(result, f)

    tf.logging.log_every_n(tf.logging.INFO, 'Processed %d/%d images...', 10, i, len(input_image_paths))

  print('Finished processing all images in data set.')


if __name__ == '__main__':
  tf.app.run()
