#!/usr/bin/python3
import tensorflow as tf

def convert_onnx_to_tflite(onnx_model_path, tflite_model_path):
    # Load the ONNX model
    onnx_model = tf.saved_model.load(onnx_model_path)

    # Convert the model to a TensorFlow Lite model
    converter = tf.lite.TFLiteConverter.from_saved_model(onnx_model)
    tflite_model = converter.convert()

    # Save the TensorFlow Lite model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

convert_onnx_to_tflite('model.onnx', 'model.tflite')
