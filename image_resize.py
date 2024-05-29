import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.vgg16 import preprocess_input

#The directory
dataset_dir = 'G:/peethon/archive (1)/Indian Food Images/Indian Food Images/adhirasam'

#target size for resizing for vgg16 that is 224 , 224
target_size = (224, 224)

#preprocessing images for subdirectories
def preprocess_images(dataset_dir, target_size):
    X = []
    for root, dirs, files in os.walk(dataset_dir):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            for img_file in os.listdir(folder_path):
                img_path = os.path.join(folder_path, img_file)
            
                img = load_img(img_path, target_size=target_size)#load the image
                # Convert image to numpy array
                img_array = img_to_array(img)
                # Preprocess the image (normalize pixel values)
                img_array = preprocess_input(img_array)
                # Append preprocessed image to list
                X.append(img_array)
    return np.array(X)

# do the preprocessing for all the folders
X = preprocess_images(dataset_dir, target_size)

# checking the sizes of the preprocessed images
print("Shape of preprocessed images array:", X.shape)
