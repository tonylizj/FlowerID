# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
from tensorflow.keras.models import Sequential, model_from_json, load_model
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout, MaxPooling2D, BatchNormalization, Activation, DepthwiseConv2D, AvgPool2D
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import backend as K


import os
import matplotlib.pyplot as plt

cwd = os.getcwd()
PATH = os.path.join(cwd, 'nonResized')

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


# %%
batch_size = 128
epochs = 10
IMG_HEIGHT = 200
IMG_WIDTH = 200


# %%
train_dir = os.path.join(PATH, 'train')
validation_dir = os.path.join(PATH, 'validation')
print(train_dir)


# %%
total_train, total_val = 0, 0

for category in os.listdir(train_dir):
    total_train += len(os.listdir(os.path.join(train_dir, category)))
for category in os.listdir(validation_dir):
    total_val += len(os.listdir(os.path.join(validation_dir, category)))

print("Training images: " + str(total_train))
print("Validation images: " + str(total_val))


# %%
train_image_generator = ImageDataGenerator(rescale=1./255,
                                           rotation_range=45,
                                           width_shift_range=.15,
                                           height_shift_range=.15,
                                           horizontal_flip=True,
                                           zoom_range=0.5)
validation_image_generator = ImageDataGenerator(rescale=1./255)

train_data_gen = train_image_generator.flow_from_directory(batch_size=batch_size,
                                                           directory=train_dir,
                                                           shuffle=True,
                                                           target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                           class_mode='categorical')

val_data_gen = validation_image_generator.flow_from_directory(batch_size=batch_size,
                                                              directory=validation_dir,
                                                              target_size=(IMG_HEIGHT, IMG_WIDTH),
                                                              class_mode='categorical')

sample_training_images, _ = next(train_data_gen)


# %%
def plotImages(images_arr):
    fig, axes = plt.subplots(1, 5, figsize=(20,20))
    axes = axes.flatten()
    for img, ax in zip( images_arr, axes):
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


plotImages(sample_training_images[:5])


# %%
'''
model = Sequential([
    Conv2D(16, 3, padding='same', activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH ,3)),
    MaxPooling2D(),
    Conv2D(32, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Conv2D(64, 3, padding='same', activation='relu'),
    MaxPooling2D(),
    Flatten(),
    Dense(512, activation='relu'),
    Dense(5, activation='softmax')
])
'''


model = Sequential()
model.add(Conv2D(32, 3, strides = 1, padding = "same", activation = "relu", input_shape = (IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(Conv2D(32, 3, strides = 1, padding = "same", activation = "relu"))

model.add(MaxPooling2D(2))

model.add(Conv2D(64, 3, strides = 1, padding = "same", activation = "relu"))
model.add(Conv2D(64, 3, strides = 1, padding = "same", activation = "relu"))

model.add(MaxPooling2D(2))

model.add(Conv2D(128, 3, strides = 1, padding = "same", activation = "relu"))
model.add(Conv2D(128, 3, strides = 1, padding = "same", activation = "relu"))

model.add(MaxPooling2D(2))

model.add(Conv2D(256, 3, strides = 1, padding = "same", activation = "relu"))
model.add(Conv2D(256, 3, strides = 1, padding = "same", activation = "relu"))

model.add(MaxPooling2D(2))

model.add(Flatten())
model.add(Dense(4096, activation = "relu"))
model.add(Dense(4096, activation = "relu"))

model.add(Dense(1000, activation = "relu"))
model.add(Dense(5, activation = "softmax"))


#model = tf.keras.models.load_model('model.h5')

'''


model = tf.keras.models.load_model("model.h5")
'''


# %%
opt = tf.keras.optimizers.Adam(lr=0.001)

model.compile(optimizer = opt,
              loss = 'categorical_crossentropy',
              metrics = ['accuracy'])



# %%
'''
K.set_value(model.optimizer.lr, 0.0001)
'''
history = model.fit_generator(
    train_data_gen,
    steps_per_epoch=total_train // batch_size,
    epochs=epochs,
    validation_data=val_data_gen,
    validation_steps=total_val // batch_size
    )


# %%
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']

loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(epochs)


# %%
plt.figure(figsize=(8, 8))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, label='Training Accuracy')
plt.plot(epochs_range, val_acc, label='Validation Accuracy')
plt.legend(loc='lower right')
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, label='Training Loss')
plt.plot(epochs_range, val_loss, label='Validation Loss')
plt.legend(loc='upper right')
plt.title('Training and Validation Loss')
plt.show()


# %%
model.summary()

model.save("model.h5")


model_json = model.to_json()

with open("model_num.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights('model_weights.h5')


