#!/usr/bin/python3


import tensorflow as tf
"""
# make a converter object from the saved tensorflow file
converter = tf.lite.TFLiteConverter.from_saved_model('model.pb')
# tell converter which type of optimization techniques to use
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

# convert the model
tf_lite_model = converter.convert()
# save the converted model
open('esrgan.tflite', 'wb').write(tf_lite_model)
"""

model = tf.saved_model.load('model.pb')
concrete_func = model.signatures[
  tf.saved_model.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
concrete_func.inputs[0].set_shape([None, 3, 100, 100])
converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
converter.optimizations = [tf.lite.Optimize.DEFAULT]
# to view the best option for optimization read documentation of tflite about optimization
# go to this link https://www.tensorflow.org/lite/guide/get_started#4_optimize_your_model_optional

# convert the model
tf_lite_model = converter.convert()
# save the converted model
open('esrgan.tflite', 'wb').write(tf_lite_model)
