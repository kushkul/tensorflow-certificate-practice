import os
import tensorflow as tf
import zipfile
import urllib.request

# -------- Aim of this Lab is to use cats and dogs dataset and see the difference between model
# -------- performance when augmentations are used vs when they are not used

print('Downloading the data')
# Downloading the dataset
urllib.request.urlretrieve('https://storage.googleapis.com/tensorflow-1-public/course2/cats_and_dogs_filtered.zip',
                           filename='data/cats_and_dogs_filtered.zip')

print('Unzipping the data')
# Unzipping the file
with zipfile.ZipFile('data/cats_and_dogs_filtered.zip', 'r') as zip_obj:
    zip_obj.extractall('data/')

print('Creating Image data generators')
# Making image data generators from the data
train_base_path = './data/cats_and_dogs_filtered/train/'
valid_base_path = './data/cats_and_dogs_filtered/validation/'

train_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)
valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255.)

train_data_gen = train_gen.flow_from_directory(train_base_path, target_size=(180, 180),
                                               class_mode='binary')
valid_data_gen = valid_gen.flow_from_directory(valid_base_path, target_size=(180, 180),
                                               class_mode='binary')


train_gen_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255., rotation_range=20,
                                                                width_shift_range=0.2, height_shift_range=0.2,
                                                                shear_range=0.2, zoom_range=0.2,
                                                                fill_mode='nearest', horizontal_flip=True)
valid_gen_aug = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1/255., rotation_range=20,
                                                                width_shift_range=0.2, height_shift_range=0.2,
                                                                shear_range=0.2, zoom_range=0.2,
                                                                fill_mode='nearest', horizontal_flip=True)

train_data_gen_aug = train_gen_aug.flow_from_directory(train_base_path, target_size=(180, 180),
                                                       class_mode='binary')
valid_data_gen_aug = valid_gen_aug.flow_from_directory(valid_base_path, target_size=(180, 180),
                                                       class_mode='binary')


def create_model():
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(16, (3,3), activation='relu', input_shape=(180,180,3)),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Conv2D(32, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D((2,2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(512, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer=tf.keras.optimizers.RMSprop(),
                  loss='binary_crossentropy', metrics=['accuracy'])

    return model


# Creating and training first model without augmentations
model = create_model()
print(model.summary())
print('Training non augmented model')

hist = model.fit(train_data_gen, epochs=10,
                 validation_data=valid_data_gen)

print('Training augmented model')

aug_model = create_model()
hist_aug = aug_model.fit(train_data_gen_aug, epochs=10,
                         validation_data=valid_data_gen_aug)


# Conclusion - In case of non augmented data, accuracy on training set is very high but on validation set is low
# compared to the training, hence there is overfitting.
# In case of augmented data, the overall training and validation accuracy is decreased, but validation accuracy is
# inline with the training accuracy, so model does not overfit.

