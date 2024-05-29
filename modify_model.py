from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16

# Load the VGG16 model but exclude the top layer, which is responsible for classification
base_model = VGG16(weights='imagenet', include_top=False)

# Define the number of classes in your dataset
NUM_CLASSES = 78  # Replace 78 with the actual number of classes in your dataset

# Freeze the base VGG16 model layers
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)  # Add a global average pooling layer
x = Dense(256, activation='relu')(x)  # Add a fully connected layer with 256 units and ReLU activation
predictions = Dense(NUM_CLASSES, activation='softmax')(x)  # Add a final output layer for classification

# Combine the base model and new layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

