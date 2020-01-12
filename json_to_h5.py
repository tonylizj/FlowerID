import tensorflow as tf
import tensorflowjs as tfjs
from tensorflow.keras.models import model_from_json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

json_load_file = open("model_num.json", "r")

loaded_model_json = json_load_file.read()
json_load_file.close()

model = model_from_json(loaded_model_json)

model.load_weights("model_weights.h5")

model.save("model.h5")
