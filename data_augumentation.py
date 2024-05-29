import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Define the directory where your training images are stored
train_dir = 'G:/peethon/archive (1)/Indian Food Images/Indian Food Images'

# Define the data augmentation parameters
datagen = ImageDataGenerator(
    rotation_range=20,      # Random rotation between 0 and 20 degrees
    width_shift_range=0.2,  # Randomly shift the width by up to 20% of the image width
    height_shift_range=0.2, # Randomly shift the height by up to 20% of the image height
    shear_range=0.2,        # Shear angle in counter-clockwise direction as radians
    zoom_range=0.2,         # Random zoom in/out by up to 20%
    horizontal_flip=True,  # Randomly flip images horizontally
    vertical_flip=True,    # Randomly flip images vertically
    fill_mode='nearest'    # Strategy for filling in newly created pixels
)

# Create a generator for loading and augmenting images from the training directory
train_generator = datagen.flow_from_directory(
    train_dir,              # Target directory
    target_size=(224, 224), # Resize images to 224x224 pixels
    batch_size=32,          # Batch size
    class_mode='categorical' # Classification mode
)

# Define the number of steps per epoch based on the number of training samples
num_train_samples = train_generator.samples
steps_per_epoch = num_train_samples // train_generator.batch_size

# Now you can use train_generator to train your model using data augmentation
