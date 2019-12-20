from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os


def predict(model):
    cwd = os.getcwd()
    labels = [i for i in os.listdir(os.path.join(cwd, 'nonResized\\train'))]
    input_dir = os.path.join(cwd, 'input')
    input_gen = ImageDataGenerator(rescale=1./255)
    input_data_gen = input_gen.flow_from_directory(batch_size=128,
                                                   directory=input_dir,
                                                   target_size=(200, 200),
                                                   class_mode='categorical')

    print(labels[model.predict(input_data_gen).argmax(axis=-1)[0]])