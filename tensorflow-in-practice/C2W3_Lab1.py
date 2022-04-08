# Aim of this file is to practice using transfer learning using InceptionNet
# on cats and dogs dataset. We will load the model first, retrain the last layers and then fine tune it

import tensorflow as tf
import urllib.request
import os

# Getting the weights to be used
urllib.request.urlretrieve('https://storage.googleapis.com/mledu-datasets/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5',
                           filename='./data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5')

# Loading the model and the weights
base_model = tf.keras.applications.inception_v3.InceptionV3(include_top=False, weights=None,
                                                            input_shape=(150, 150, 3))

local_weights = './data/inception_v3_weights_tf_dim_ordering_tf_kernels_notop.h5'
base_model.load_weights(local_weights)

# Freezing the layers for training
for layer in base_model.layers:
    layer.trainable = False

print(base_model.summary())

# Using mized-7 as the output layer, ignoring all the layers after it
last_layer = base_model.get_layer('mixed7')
print('Shape of output mixed7 layer is {}'.format(last_layer.output_shape))
last_output = last_layer.output

# Adding dense layers to the model
x = tf.keras.layers.Flatten()(last_output)
x = tf.keras.layers.Dense(1024, activation='relu')(x)
x = tf.keras.layers.Dropout(0.2)(x)
x = tf.keras.layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.Model(base_model.input, x)

model.compile(optimizer=tf.keras.optimizers.RMSprop(),
              loss='binary_crossentropy', metrics=['accuracy'])

print(model.summary())


# Preparing the inputs
base_data_dir = './data/cats_and_dogs_filtered/'
train_data_dir = os.path.join(base_data_dir, 'train')
valid_data_dir = os.path.join(base_data_dir, 'validation')


train_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,rotation_range=20,
                                                                 width_shift_range=0.2, height_shift_range=0.2,
                                                                 shear_range=0.2, zoom_range=0.2,
                                                                 horizontal_flip=True, fill_mode='nearest')

valid_data_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.,rotation_range=20,
                                                                 width_shift_range=0.2, height_shift_range=0.2,
                                                                 shear_range=0.2, zoom_range=0.2,
                                                                 horizontal_flip=True, fill_mode='nearest')

train_image_gen = train_data_gen.flow_from_directory(train_data_dir,
                                                     target_size=(150, 150), class_mode='binary',
                                                     batch_size=20)
valid_image_gen = valid_data_gen.flow_from_directory(valid_data_dir,
                                                     target_size=(150, 150), class_mode='binary',
                                                     batch_size=20)


# Training the model

hist = model.fit(train_image_gen, epochs=10,
                 validation_data=valid_image_gen, validation_batch_size=50)










