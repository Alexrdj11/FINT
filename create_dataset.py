import os
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from PIL import Image

# reading the nutritional data from the nutrition.txt file
def read_nutritional_info(folder_path):
    nutrition_file = os.path.join(folder_path, 'nutrition.txt')
    if os.path.exists(nutrition_file):
        with open(nutrition_file, 'r') as file:
            lines = file.readlines()
            nutritional_info = {}
            for line in lines:
                key, value = line.strip().split(': ')
                nutritional_info[key] = value
            return nutritional_info
    else:
        return None

# creating a dataset
def create_dataset(dataset_folder):
    data = []
    for food_folder in os.listdir(dataset_folder):
        food_folder_path = os.path.join(dataset_folder, food_folder)
        if os.path.isdir(food_folder_path):
            for image_file in os.listdir(food_folder_path):
                if image_file.endswith('.jpg'):
                    image_path = os.path.join(food_folder_path, image_file)
                    nutritional_info = read_nutritional_info(food_folder_path)
                    if nutritional_info:
                        data.append([image_path, nutritional_info['Calories'], 
                                     nutritional_info['Protein'], nutritional_info['Fat'], 
                                     nutritional_info['Carbohydrates']])
    df = pd.DataFrame(data, columns=['Image Path', 'Calories', 'Protein', 'Fat', 'Carbohydrates'])
    return df

# lets read image and nutritional info and also resize and adjust the image size and pixel values 
def read_image_and_nutritional_info(image_path, calories, protein, fat, carbohydrates):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (224, 224))  
    img = tf.cast(img, tf.float32) / 255.0  
    
    # Converting nutritional info to tensor
    nutritional_info = tf.stack([calories, protein, fat, carbohydrates])
    
    return img, nutritional_info

# Step 1: Defining the dataset folder
dataset_folder = 'path/to/dataset'

# Step 2: Creating the dataset
dataset = create_dataset(dataset_folder)
print(dataset)

# Step 3: Define data augmentation and preprocessing for images
datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

# Step 4: Create TensorFlow dataset
tf_dataset = tf.data.Dataset.from_tensor_slices((
    dataset['Image Path'].values,
    dataset['Calories'].values,
    dataset['Protein'].values,
    dataset['Fat'].values,
    dataset['Carbohydrates'].values
))

# Step 5: Mapping the function to read image and nutritional info
tf_dataset = tf_dataset.map(read_image_and_nutritional_info)

# Step 6: Apply data augmentation
tf_dataset = tf_dataset.map(lambda img, nutritional_info: (datagen.random_transform(img), nutritional_info))

# Step 7: Batch and prefetch
tf_dataset = tf_dataset.batch(32).prefetch(tf.data.experimental.AUTOTUNE)

# Example usage:
for images, nutritional_info in tf_dataset:
    # Training code goes here
    print(images.shape)
    print(nutritional_info)
