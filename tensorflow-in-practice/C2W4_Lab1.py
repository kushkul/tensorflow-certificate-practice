# Aim - Build multi class classification using hands images in Rock, Paper, and Scissors poses

import tensorflow as tf
import os
import zipfile
import urllib.request


# Downlaoding and extracting the dataset
#urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow-1-public/course2/week4/rps.zip',
#                            filename='data/rps.zip')

#with zipfile.ZipFile('data/rps.zip','r') as zip_obj:
#    zip_obj.extractall('data/')


# Image data generators
data_dir = './data/rps/'

data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255., rotation_range=20,
                                                           width_shift_range=0.2, height_shift_range=0.2,
                                                           shear_range=0.2, zoom_range=0.2,
                                                           fill_mode='nearest', horizontal_flip='True')

image_data_gen = data_gen.flow_from_directory(data_dir, target_size=(150, 150),
                                              class_mode='categorical')


# Building the model

model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(150, 150, 3)),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
    tf.keras.layers.MaxPooling2D((2,2)),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(512, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='categorical_crossentropy', metrics=['accuracy'])

print(model.summary())

print('Training the model')

model.fit(image_data_gen, epochs=10)
