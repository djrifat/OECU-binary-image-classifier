# Author: @DJrif
# Image classification script to classify
# archaeological images from old Japanese pottery patterns

import os, random
import numpy as np

# Importing Keras libraries and packages
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, GlobalMaxPooling2D, Input
from keras.layers import Activation, Dropout, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator, image
from keras.applications import VGG16
from keras import backend as K

# Importing matplotlib for basic visualization
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

# Image dimension
img_width, img_height, img_channel = 224, 224, 3

# Set size for different datasets
train_size, validation_size, test_size = 60, 40, 40

# Deeplearning model variables
batch_size = 32
epochs = 150

# Set directories for train, test and validation image paths
base_dir = 'data'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'validation')
test_dir = os.path.join(base_dir, 'test')

# Train directory
train_LR_dir = os.path.join(train_dir, 'LR')
train_RL_dir = os.path.join(train_dir, 'RL')

# Test directory
test_LR_dir = os.path.join(test_dir, 'LR')
test_RL_dir = os.path.join(test_dir, 'RL')

# Display imahges for validation check
def show_pictures(path):
    random_img = random.choice(os.listdir(path))
    img_path = os.path.join(path, random_img)

    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)  # Image data encoded as integers in the 0–255 range
    img_tensor /= 255.                    # Normalize to [0,1] for plt.imshow application
    plt.imshow(img_tensor)
    plt.show()

#for i in range(0,2):
    #show_pictures(train_LRR_dir)
    #show_pictures(train_RLL_dir)

# Check and select the backend that Keras uses
if K.image_data_format() == 'channels_first':
  input_shape = (img_channel, img_width, img_height)
else:
  input_shape = (img_width, img_height, img_channel)


# TODO: make into functions
# Build VGG16 network
# Set include_top to False to freeze VGG16 layers and weigths don't get updated
conv_base = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, img_channel))

# Check architecture
conv_base.summary()

# Data generator
datagen = ImageDataGenerator(
  rescale=1. / 255,         # Rescale images to reduce training size
  rotation_range=360,       # Apply random rotation
  width_shift_range=0.2,    # Apply random width shift
  height_shift_range=0.2,   # Apply random height shift
  shear_range=0.2,          # Apply random changes in the shear range
  zoom_range=0.2,           # Apply random zoom
  horizontal_flip=True
)

# Extract features
def extract_features(directory, sample_count):

  # Must be equal to the output of the convolutional base
  features = np.zeros(shape=(sample_count, 7, 7, 512))
  labels = np.zeros(shape=(sample_count))

  # Preprocess  data
  generator = datagen.flow_from_directory(
    directory,
    target_size = (img_width, img_height),
    batch_size = batch_size,
    class_mode = 'binary'
  )

  # Pass data through convolutional base
  i = 0
  for inputs_batch, labels_batch in generator:
      features_batch = conv_base.predict(inputs_batch)
      features[i * batch_size: (i + 1) * batch_size] = features_batch
      labels[i * batch_size: (i + 1) * batch_size] = labels_batch
      i += 1
      if i * batch_size >= sample_count:
          break
  return features, labels

# Extract features
train_features, train_labels = extract_features(train_dir, train_size)
validation_features, validation_labels = extract_features(validation_dir, validation_size)
test_features, test_labels = extract_features(test_dir, test_size)

# Debug
#print(train_features)
#print(train_labels)

# Create fully connected layers to put on top of VGG16 convolutional model
model = Sequential()
#model.add(GlobalAveragePooling2D(input_shape = (7,7,512)))
#model.add(GlobalMaxPooling2D(input_shape = (7,7,512)))
model.add(Flatten(input_shape = (7,7,512)))
model.add(Dense(256, input_dim = (7*7*512)))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

# Summary of fully connected layers
model.summary()

# Compilde model
model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Fit model
history = model.fit(
  train_features, train_labels,
  epochs=epochs,
  batch_size=batch_size,
  validation_data=(validation_features, validation_labels)
)

# Save entire model as HDf5 file
# Allows to save the entirety of the model to be saved to a single file
# Saved model can be reinstantiated this way
model.save('weights/LRR_RLL_fcl.h5')

# Plot results
def plot_result():

  # create variables for accuracy and loss
  acc = history.history['acc']
  val_acc = history.history['val_acc']
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  epochs = range(1, len(acc)+1)

  # Print average accuracy and loss from training and validation
  print("Training accuracy: ", np.mean(acc))
  print("Validation accuracy: ", np.mean(val_acc))
  print("Training loss:", np.mean(loss))
  print("Validation loss:", np.mean(val_loss))

  # Plot training and validation accuracy
  plt.plot(epochs, acc, 'bo', label='Training accuracy')
  plt.plot(epochs, val_acc, 'b', label='Validation accuracy')
  plt.title('Training and validation accuracy')
  plt.legend()

  plt.figure()

  # Plot training and validation loss
  plt.plot(epochs, loss, 'bo', label='Training loss')
  plt.plot(epochs, val_loss, 'b', label='Validation loss')
  plt.title('Training and validation loss')
  plt.legend()

  plt.show()

plot_result()

# Visualizes predictions
def visualize_predicttion(classifier, n_cases):

  # Loop through images
  for i in range(0,n_cases):

    # Set path for test images
    path = random.choice([test_LR_dir, test_RL_dir])
    #path = 'data/RandomLRR.jpg'

    # Get pictures
    random_img = random.choice(os.listdir(path))
    img_path = os.path.join(path, random_img)
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_tensor = image.img_to_array(img)    # Image data encoded as integers in the 0-255 range
    img_tensor /= 255.                      # Normalize to [0,1] for matplotlib application

    # Extract feature
    features = conv_base.predict(img_tensor.reshape(1, img_width, img_height, img_channel))

    # Make prediction
    try:
      prediction = classifier.predict(features)
    except:
      prediction = classifier.predict(features.reshape(1, 7*7*512))

    # Show image with prediction
    plt.title(random_img)
    plt.imshow(img_tensor)
    plt.show()

    # Write prediction
    if prediction < 0.5:
      print('LR')
    else:
      print('RL')


# Visualize predictions
visualize_predicttion(model, 4)
