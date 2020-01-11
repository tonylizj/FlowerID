import tensorflow as tf
from tensorflow.keras.models import model_from_json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

#model = tf.keras.models.load_model('model.h5')

json_load_file = open("model_num.json", "r")

loaded_model_json = json_load_file.read()
json_load_file.close()

model = model_from_json(loaded_model_json)

model.load_weights("model_weights.h5")

converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.post_training_quantize = True
converter.inference_type = tf.uint8
converter.default_ranges_stats = [0, 255]
converter.quantized_input_stats = {"conv2d_input": [128, 128]}
converter.optimizations = [tf.lite.Optimize.OPTIMIZE_FOR_SIZE]

tflite_model = converter.convert()

open("tflite_model.tflite", "wb").write(tflite_model)
