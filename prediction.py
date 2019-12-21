from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os
import time
import tensorflow as tf

from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession

config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

cwd = os.getcwd()
model = tf.keras.models.load_model("model.h5")
label_dir = os.path.join(cwd, 'nonResized\\train')
input_dir = os.path.join(cwd, 'input')


def predict(model, label_dir, input_dir):
    labels = [i for i in os.listdir(label_dir)]
    input_gen = ImageDataGenerator(rescale=1. / 255)
    input_data_gen = input_gen.flow_from_directory(batch_size=1,
                                                   directory=input_dir,
                                                   target_size=(200, 200),
                                                   class_mode='categorical',
                                                   shuffle=False)

    predictions = model.predict_generator(input_data_gen)
    pred_labels = predictions.argmax(axis=-1)
    filenames = input_data_gen.filenames
    for i in range(len(pred_labels)):
        print("File " + filenames[i] + " is species: " + str(labels[pred_labels[i]]) + " - Confidence: " +
              str(predictions[i][pred_labels[i]]))


predict(model, label_dir, input_dir)
