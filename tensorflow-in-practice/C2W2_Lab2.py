# -------- Aim of this Lab is to use horses vs humans dataset and see the difference between model
# -------- performance when augmentations are used vs when augmentation is not used

import tensorflow as tf
import os
import zipfile
import urllib.request

print('Downloading the dataset')
urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow-1-public/course2/week3/horse-or-human.zip',
                           filename='data/horse-or-human.zip')

urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow-1-public/course2/week3/validation-horse-or-human.zip',
                           filename='data/validation-horse-or-human.zip')

print('Unzipping the content')
print('Unzipping training data')
with zipfile.ZipFile('data/horse-or-human.zip', 'r') as zip_obj:
    zip_obj.extractall('data/horse-or-human')
print('Unzipping validation data')
with zipfile.ZipFile('data/validation-horse-or-human.zip', 'r') as zip_obj:
    zip_obj.extractall('data/validation-horse-or-human')

train_base_path = './data/horse-or-human/'
valid_base_path = './data/validation-horse-or-human'


print('Creating Image data generators with augmentation')

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255., rotation_range=20,
                                                            width_shift_range=0.2, height_shift_range=0.2,
                                                            shear_range=0.2, zoom_range=0.2,
                                                            horizontal_flip=True, fill_mode='nearest')

valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255., rotation_range=20,
                                                            width_shift_range=0.2, height_shift_range=0.2,
                                                            shear_range=0.2, zoom_range=0.2,
                                                            horizontal_flip=True, fill_mode='nearest')

print('Creating train image data generator')
train_data_gen = train_gen.flow_from_directory(train_base_path, target_size=(180, 180),
                                               class_mode='binary')
print('Creating validation Image data Generator')
valid_data_gen = valid_gen.flow_from_directory(valid_base_path, target_size=(180, 180),
                                               class_mode='binary')


print('Creating the model')

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Conv2D(64, (3, 3), activation='relu', input_shape=(180, 180, 3)),
    tf.keras.layers.MaxPooling2D((2, 2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='binary_crossentropy', metrics=['accuracy'])

print('Training the model')
hist = model.fit(train_data_gen, epochs=10,
                 validation_data=valid_data_gen)

