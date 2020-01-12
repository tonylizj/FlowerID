import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.models import model_from_json
import os
import numpy as np
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

model = tf.keras.models.load_model("model.h5")

'''
json_load_file = open("model_num.json", "r")

loaded_model_json = json_load_file.read()
json_load_file.close()

model = model_from_json(loaded_model_json)

model.load_weights("model_weights.h5")
'''

tfjs.converters.save_keras_model(model, "tfjs_model", quantization_dtype=np.uint8)
