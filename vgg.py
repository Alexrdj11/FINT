import tensorflow as tf

# Load the pre-trained VGG16 model
vgg16_model = tf.keras.applications.VGG16(weights='imagenet', include_top=True)

# Display a summary of the VGG16 model architecture
vgg16_model.summary()
