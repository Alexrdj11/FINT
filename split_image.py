import os
import shutil

# Define the directory where your dataset is stored
dataset_dir = 'G:/peethon/archive (1)/Indian Food Images/Indian Food Images'



# Function to split images into training, validation, and testing directories
def split_images(dataset_dir, train_ratio, val_ratio, test_ratio):
    for root, dirs, files in os.walk(dataset_dir):
        for folder in dirs:
            folder_path = os.path.join(root, folder)
            images = os.listdir(folder_path)
            num_images = len(images)
            num_train = int(train_ratio * num_images)
            num_val = int(val_ratio * num_images)
            num_test = int(test_ratio * num_images)
            
            train_images = images[:num_train]
            val_images = images[num_train:num_train + num_val]
            test_images = images[num_train + num_val:]
            
            # Create directories for training, validation, and testing
            train_dir = os.path.join(folder_path, 'training')
            val_dir = os.path.join(folder_path, 'validation')
            test_dir = os.path.join(folder_path, 'testing')
            
            os.makedirs(train_dir, exist_ok=True)
            os.makedirs(val_dir, exist_ok=True)
            os.makedirs(test_dir, exist_ok=True)
            
            # Move images to corresponding directories
            for img in train_images:
                src = os.path.join(folder_path, img)
                dst = os.path.join(train_dir, img)
                shutil.move(src, dst)
                
            for img in val_images:
                src = os.path.join(folder_path, img)
                dst = os.path.join(val_dir, img)
                shutil.move(src, dst)
                
            for img in test_images:
                src = os.path.join(folder_path, img)
                dst = os.path.join(test_dir, img)
                shutil.move(src, dst)

# Split images into training, validation, and testing directories
split_images(dataset_dir, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
