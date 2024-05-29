import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.callbacks import ModelCheckpoint

# Define the directory containing your organized dataset
dataset_dir = 'G:/peethon/archive (1)/Indian Food Images/Indian Food Images'

# Define batch size
batch_size = 32

# Create TensorFlow datasets with resizing
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='categorical',  # Change this to 'categorical'
    image_size=(224, 224),  # Resize images to (224, 224)
    batch_size=batch_size,
    validation_split=0.2,  # Adjust validation split as needed
    subset='training',
    seed=123,
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='categorical',  # Change this to 'categorical'
    image_size=(224, 224),  # Resize images to (224, 224)
    batch_size=batch_size,
    validation_split=0.2,  # Adjust validation split as needed
    subset='validation',
    seed=123,
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='categorical',  # Change this to 'categorical'
    image_size=(224, 224),  # Resize images to (224, 224)
    batch_size=batch_size,
)

# Optionally, you can prefetch data for better performance
train_dataset = train_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
val_dataset = val_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
test_dataset = test_dataset.prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

# Load the pre-trained VGG16 model without the top (fully connected) layers
base_model = tf.keras.applications.VGG16(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze the base VGG16 model layers
for layer in base_model.layers:
    layer.trainable = False

# Add new layers on top of the base model
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(256, activation='relu')(x)
predictions = Dense(78, activation='softmax')(x)

# Combine the base model and new layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Define the ModelCheckpoint callback
checkpoint_filepath = 'G:\mark1\checkpoints.weights.h5'
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',  # You can change the monitored metric as needed
    mode='max',  # You can change the mode as needed (e.g., 'max' for accuracy, 'min' for loss)
    save_best_only=True
)

# Train the model with the checkpoint callback
history = model.fit(
    train_dataset,
    epochs=15,
    validation_data=val_dataset,
    callbacks=[checkpoint_callback]  # Pass the checkpoint callback here
)

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
