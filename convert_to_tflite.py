import tensorflow as tf

from absl import app, flags, logging
from absl.flags import FLAGS

from yolov3_tf2.models import YoloV3, YoloV3Tiny

flags.DEFINE_string('weights', './checkpoints/yolov3-tiny.tf', 'path to weights file')
flags.DEFINE_boolean('tiny', True, 'yolov3 or yolov3-tiny')
flags.DEFINE_integer('num_classes', 80, 'number of classes in the model')
flags.DEFINE_integer('input_size', 416, 'image size')


# https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/delegates/flex/whitelisted_flex_ops.cc
# https://www.tensorflow.org/api_docs/python/tf/image/non_max_suppression
# https://www.tensorflow.org/s/results/?q=combined_non_max_suppression

def main(_argv):
    if FLAGS.tiny:
        yolo = YoloV3Tiny(FLAGS.input_size, classes=FLAGS.num_classes)
    else:
        yolo = YoloV3(FLAGS.input_size, classes=FLAGS.num_classes)

    yolo.load_weights(FLAGS.weights).expect_partial()
    logging.info('weights loaded')

    path = 'tmp/yolov3.h5'
    out_path = 'tmp/yolov3.tflite'

    if FLAGS.tiny:
        path = 'tmp/yolov3-tiny.h5'
        out_path = 'tmp/yolov3-tiny.tflite'

    yolo.save(path)

    logging.info(f'keras model saved to location {path}')

    converter = tf.lite.TFLiteConverter.from_keras_model(yolo)
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS
    ]
    converter.experimental_new_converter = True
    converter.experimental_new_quantizer = True
    converter.allow_custom_ops = True

    tflite_model = converter.convert()
    logging.info(f'model converted')

    open(out_path, 'wb').write(tflite_model)

    logging.info(f'tflite model saved to location {out_path}')


if __name__ == '__main__':
    try:
        app.run(main)
    except SystemExit:
        pass