from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import pandas as pd
from sklearn.model.selection import train_test_split

import os
import numpy as np
import matplotlib.pyplot as plt

file_path = 'C:\Users\Daniel\Documents\GitHub\\flower-id\\flowers'
_path = tf.keras.utils.get_file('flowers', origin=file_path, extract=False)
PATH = os.path.join(os.path.dirname(_path), 'flowers')



train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')

