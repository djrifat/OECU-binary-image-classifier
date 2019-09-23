from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

import matplotlib.pyplot as plt
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# dimensions of our images.
img_width, img_height = 150, 150

# Variables for model training
train_data_dir = 'data/CatsVsDogs/train'
validation_data_dir = 'data/CatsVsDogs/validation'
nb_train_samples = 2000
nb_validation_samples = 800
epochs = 100
batch_size = 16

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

# Deep Learning model layers
# Model outputs 3D feature maps (height, width, features)
model = Sequential()

model.add(Conv2D(32, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())      # this converts our 3D feature maps to 1D feature vectors
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Compile model, uses binary-crossebtropy loss to train model
model.compile(loss='binary_crossentropy',
              optimizer='rmsprop',
              metrics=['accuracy'])

# Image augmentation configuration used for training
# TODO extand image augmentation
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Image augmentation configuration used for testing
# Only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

# Generator that will read pictures found in
# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Generator for validation data
validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary')

# Train Deep Learning model
model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

# save weights
model.save_weights('first_try.h5')

# Get training and test loss
#training_loss = history.history['val_acc']

# Plot model accuracy
#plt.figure(figsize=(10, 6))
#plt.axis((-10,310,0.965,0.983))

#plt.plot(training_loss)
#plt.title('val.accuracy')
#plt.ylabel('accuracy')
#plt.xlabel('epoch')
#plt.legend(['rmsprop'], loc = 'lower_right')

#plt.show()
