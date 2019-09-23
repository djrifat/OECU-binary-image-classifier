# Author: @DJrif
# Image classification script to classify
# archaeological images from old Japanese pottery patterns

import os, random, shutil
import numpy as np

# Set all directories for image classifier
# If directories don't exists, they will be created
def create_dir(base_dir, train_dir, validation_dir, test_dir):

  # Set base directories
  base_dir = base_dir
  train_dir = os.path.join(base_dir, train_dir)
  validation_dir = os.path.join(base_dir, validation_dir)
  test_dir = os.path.join(base_dir, test_dir)

  # Create base directory if it doesnt exists
  if not os.path.exists(base_dir):
    os.mkdir(base_dir)

  # Create base directories for train, validation and test
  # If not exists, they will be created
  if not os.path.exists(train_dir):       # Train
    os.mkdir(train_dir)
  if not os.path.exists(validation_dir):  # Validation
    os.mkdir(validation_dir)
  if not os.path.exists(test_dir):        # Test
    os.mkdir(test_dir)

  # Create LRR and RLL directories for train and test
  # Train directories
  train_LRR_dir = os.path.join(train_dir, 'LRR')
  if not os.path.exists(train_LRR_dir):
    os.mkdir(train_LRR_dir)

  train_RLL_dir = os.path.join(train_dir, 'RLL')
  if not os.path.exists(train_RLL_dir):
    os.mkdir(train_RLL_dir)

  # Test directories
  test_LRR_dir = os.path.join(test_dir, 'LRR')
  if not os.path.exists(test_LRR_dir):
    os.mkdir(test_LRR_dir)

  test_RLL_dir = os.path.join(test_dir, 'RLL')
  if not os.path.exists(test_RLL_dir):
    os.mkdir(test_RLL_dir)

  # Return tulpe with directories
  return train_LRR_dir, train_RLL_dir, test_LRR_dir, test_RLL_dir, validation_dir, train_dir, test_dir

# Assign returned tulpe from create_dir() function
train_LRR_dir, train_RLL_dir, test_LRR_dir, test_RLL_dir, validation_dir, train_dir, test_dir = create_dir('data', 'train', 'test', 'validation')

