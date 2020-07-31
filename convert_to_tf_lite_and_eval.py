from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation, DepthwiseConv2D, AvgPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K
import numpy as np


import os
import matplotlib.pyplot as plt

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
converter.optimizations = [tf.lite.Optimize.DEFAULT]

tflite_model = converter.convert()

open("tflite_model.tflite", "wb").write(tflite_model)

# Load TFLite model and allocate tensors.
interpreter = tf.lite.Interpreter(model_path="tflite_model.tflite")
interpreter.allocate_tensors()# Get input and output tensors.
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()# Test model on some input data.
input_shape = input_details[0]['shape']
acc=0

batch_size = 128
IMG_HEIGHT = 200
IMG_WIDTH = 200

cwd = os.getcwd()
PATH = os.path.join(cwd, 'nonResized')
validation_dir = os.path.join(PATH, 'validation')
validation_image_generator = ImageDataGenerator(rescale=1./255)
val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

for i in range(len(val_data_gen)):
 input_data = val_data_gen[i]
 interpreter.set_tensor(input_details[0]['index'], input_data)
 interpreter.invoke()
 output_data = interpreter.get_tensor(output_details[0]['index'])
 if(np.argmax(output_data) == np.argmax(y_test[i])):
 acc+=1
acc = acc/len(x_test)
print(acc*100)