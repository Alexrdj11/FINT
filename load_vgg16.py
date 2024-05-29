import tensorflow as tf

# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = tf.keras.applications.VGG16(weights='imagenet',  # Load weights pre-trained on ImageNet
                                         include_top=False,   # Exclude the top (fully connected) layers
                                         input_shape=(224, 224, 3))  # Specify input shape of images
