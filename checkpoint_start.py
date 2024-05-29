
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

# Define the directory containing your organized dataset
dataset_dir = 'G:/peethon/archive (1)/Indian Food Images/Indian Food Images'

# Define batch size
batch_size = 32

# Create TensorFlow datasets
train_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=(224, 224),  # Ensure all images are resized to the same dimensions
    validation_split=0.2,  # Adjust validation split as needed
    subset='training',
    seed=123,
)

val_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=(224, 224),  # Ensure all images are resized to the same dimensions
    validation_split=0.2,  # Adjust validation split as needed
    subset='validation',
    seed=123,
)

test_dataset = tf.keras.preprocessing.image_dataset_from_directory(
    dataset_dir,
    labels='inferred',
    label_mode='int',
    batch_size=batch_size,
    image_size=(224, 224),  # Ensure all images are resized to the same dimensions
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
x = Dropout(0.5)(x)  # Add Dropout layer
x = Dense(256, activation='relu')(x)
x = Dropout(0.5)(x)  # Add another Dropout layer
predictions = Dense(78, activation='softmax')(x)

# Combine the base model and new layers into a new model
model = Model(inputs=base_model.input, outputs=predictions)

# Load the weights
model.load_weights('G:/mark1/checkpoint/model_weights.weights.h5')

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model checkpoint callback
checkpoint_callback = ModelCheckpoint(
    filepath='G:/mark1/checkpoint/model_weights.weights.h5',
    save_weights_only=True,
    monitor='val_accuracy',
    mode='max',
    save_best_only=True
)

# Continue training the model
history = model.fit(
    train_dataset,
    initial_epoch=6,  # Start from the last saved epoch + 1
    epochs=25,  # Continue training for more epochs
    validation_data=val_dataset,
    callbacks=[early_stopping, checkpoint_callback]
)

# Save the final model
model.save('final_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)
