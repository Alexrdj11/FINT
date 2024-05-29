import os
from PIL import Image

folder_path = 'G:/peethon/archive (1)/Indian Food Images/Indian Food Images/adhirasam'
target_size = (224, 224)

def check_image_size(image_path):
    try:
        with Image.open(image_path) as img:
            if img.size != target_size:
                print(f"Image {image_path} is {img.size}, not {target_size}.")
                return False
        return True
    except Exception as e:
        print(f"Error processing image {image_path}: {e}")
        return False

all_correct_size = True
for root, dirs, files in os.walk(folder_path):
    for file in files:
        if file.endswith(('jpg', 'jpeg', 'png')):
            image_path = os.path.join(root, file)
            if not check_image_size(image_path):
                all_correct_size = False

if all_correct_size:
    print("All images are the correct size.")
else:
    print("Some images are not the correct size. Please check the logs above.")
