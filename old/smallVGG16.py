# Author: @DJrif
# Image classification script to classify
# archaeological images from old Japanese pottery patterns

import os
import numpy as np

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense
from keras import backend as K

# Importing Keras libraries for data augmentation
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image

# Importing matplotlib for basic visualization
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'


# Deep Learning image classifier with small VGG16 architecture
def cnnClassifier():

  # Image dimension
  img_width, img_height = 512, 512
  imgChannel = 3

  # Deeplearning model variables
  train_data_dir = 'data/train'
  validation_data_dir = 'data/validation'
  #nb_train_samples = 70
  #nv_test_samples = 30
  epochs = 10
  batch_size = 32

  # Select which backend Keras uses
  if K.image_data_format() == 'channels_first':
    input_shape = (imgChannel, img_width, img_height)
  else:
    input_shape = (img_width, img_height, imgChannel)

  # TODO expirement with neurons and layers
  # Initialize the CNN
  model = Sequential()

  # Block 1
  model.add(Conv2D(32, (3,3), input_shape = input_shape, padding = 'same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

  # Block 2
  model.add(Conv2D(32, (3,3), input_shape = input_shape, padding = 'same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

  # Block 3
  model.add(Conv2D(64, (3,3), input_shape = input_shape, padding = 'same'))
  model.add(Activation('relu'))
  model.add(MaxPooling2D(pool_size = (2,2), strides = (2,2)))

  # Classification block
  model.add(Flatten())
  model.add(Dense(64))
  model.add(Activation('relu'))
  model.add(Dropout(0.5))
  model.add(Dense(1))
  model.add(Activation('sigmoid'))

  # Compile model, uses binary-crossebtropy loss to train model
  model.compile(loss='binary_crossentropy',
              optimizer='adam',
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

  # Generator reads pictures found in
  # subfolers of 'data/train', and indefinitely generate
  # batches of augmented image data
  # Train set only
  train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary'
  )

  # Image augmentation configuration used for testing
  test_datagen = ImageDataGenerator(
    rescale=1./255            # Rescale images to reduce training size
  )

  # Generator for validation data
  test_generator = test_datagen.flow_from_directory(
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
    validation_data = test_generator,
    validation_steps = 10
  )

  # list all data in history
  print(history.history.keys())
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

  print(model.summary())

  test_image = image.load_img('data/train/RLL/random.jpg', target_size=(512,512))
  test_image = image.img_to_array(test_image)
  test_image = np.expand_dims(test_image, axis=0)
  result = model.predict(test_image)
  train_generator.class_indices
  if result[0][0] >= 0.5:
    prediction = "LRR"
  else:
    prediction = "RLL"
  print(prediction)


cnnClassifier()





