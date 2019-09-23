# Author: @DJrif
# Image classification script to classify
# archaeological images from old Japanese pottery patterns

import os
import numpy as np

# Importing Keras libraries and packages
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import applications, optimizers
from keras import backend as K

# Importing Keras libraries for data augmentation
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Importing matplotlib for basic visualization
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Path to weight files
weights_path = '../keras/examples/vgg16_weights.h5'
top_model_weights_path = 'fc_model.h5'

#Image dimension
img_width, img_height = 512, 512
imgChannel = 3

# Deeplearning model variables
train_data_dir = 'data/train'
validation_data_dir = 'data/validation'
nb_train_samples = 70
nv_test_samples = 30
epochs = 10
batch_size = 16

# Build VGG16 network
input_tensor = Input(shape=(512,512,3))
model = applications.VGG16(weights='imagenet',include_top= False,input_tensor=input_tensor)
print("Model loaded. ")

# Classifier model to put on top of the convolutional network
top_model = Sequential()
top_model.add(Flatten(input_shape=model.output_shape[1:]))
top_model.add(Dense(256, activation='relu'))
top_model.add(Dropout(0.5))
top_model.add(Dense(1, activation='sigmoid'))

top_model.load_weights(top_model_weights_path)

# Add model on top of convolutional base
model.add(top_model)

# Set first 25 layers (up to the last convolutional block)
# to non-trainable (weights will not be updated)
for layer in model.layers[:25]:
  layer.trainable = False

# Compile model with SGG optimizer and slow learning rate
model.compile(loss='binary_crossentropy',
              optimizer = optimizers.SGD(lr = 1e-4, momentum = 0.9),
              metrics=['accuracy'])

  # Image augmentation configuration used for training
train_datagen = ImageDataGenerator(
  rescale=1. / 255,         # Rescale images to reduce training size
  rotation_range=40,        # Apply random rotation
  width_shift_range=0.2,    # Apply random width shift
  height_shift_range=0.2,   # Apply random height shift
  shear_range=0.2,          # Apply random changes in the shear range
  zoom_range=0.2,           # Apply random zoom
  horizontal_flip=True,     # Horizontal flip images
  fill_mode='nearest'
)

# Image augmentation configuration used for testing
test_datagen = ImageDataGenerator(
  rescale=1./255            # Rescale images to reduce training size
)

# subfolers of 'data/train', and indefinitely generate
# batches of augmented image data
# Train set only
train_generator = train_datagen.flow_from_directory(
  train_data_dir,
  target_size=(img_width, img_height),
  batch_size=batch_size,
  class_mode='binary'
)

# Generator for validation data
validation_generator = test_datagen.flow_from_directory(
  validation_data_dir,
  target_size=(img_width, img_height),
  batch_size=batch_size,
  class_mode='binary'
)

# Fit the model (train Deep Learning model)
# TODO change variables
history = model.fit_generator(
  train_generator,
  steps_per_epoch = 10,
  epochs = epochs,
  validation_data = validation_generator,
  validation_steps = 10
)
