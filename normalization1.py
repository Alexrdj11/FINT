import os
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.preprocessing.image import smart_resize

# Defining the target size for resizing images
target_size = (224, 224)

resized_images_dir = 'G:/peethon/archive (1)/Indian Food Images/Indian Food Images'

# Function to normalize images
def normalize_images(resized_images_dir):
    X = []
    for root, dirs, files in os.walk(resized_images_dir):
        for file in files:
            if file.endswith(".jpg") or file.endswith(".png"):
                img_path = os.path.join(root, file)
                # Loading image
                img = load_img(img_path)
                # Resizing image to target size
                img_resized = smart_resize(img, target_size)
                # Converting image to numpy array
                img_array = img_to_array(img_resized)
                # Normalizing pixel values to range [0, 1]
                normalized_img_array = img_array / 255.0
                # Appending normalized image to list
                X.append(normalized_img_array)
    return np.array(X)

# Normalizing images
X_normalized = normalize_images(resized_images_dir)

# Check the shape of the normalized images array
print("Shape of normalized images array:", X_normalized.shape)
