from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications import VGG16

# Load the VGG16 model but exclude the top layer, which is responsible for classification
base_model = VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Define the number of classes in your dataset
NUM_CLASSES = 78  # Replace 78 with the actual number of classes in your dataset

# Freeze the base VGG16 model layers
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(NUM_CLASSES, activation='softmax')(x)

# Combine the base model and new layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
