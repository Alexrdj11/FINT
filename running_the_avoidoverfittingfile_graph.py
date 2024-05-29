import tensorflow as tf
from tensorflow.keras.preprocessing import image
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
import matplotlib.pyplot as plt

# Define the directory containing your organized dataset
dataset_dir = 'G:/peethon/archive (1)/Indian Food Images/Indian Food Images'

def preprocess_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))  # Resize to 224x224
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = tf.keras.applications.vgg16.preprocess_input(img_array)  # Normalize the image
    return img_array

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

# Compile the model
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Early stopping callback
early_stopping = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

# Model checkpoint callback with corrected file path
checkpoint_filepath = 'checkpoint/model_weights.weights.h5'  # Corrected file path
checkpoint_callback = ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='val_accuracy',  # You can change the monitored metric as needed
    mode='max',  # You can change the mode as needed (e.g., 'max' for accuracy, 'min' for loss)
    save_best_only=True
)

# Train the model
history = model.fit(
    train_dataset,
    epochs=50,  # Increased epochs, but training will stop early if no improvement
    validation_data=val_dataset,
    callbacks=[early_stopping, checkpoint_callback]  # Include both callbacks
)

# Save the entire model
model.save('G:/mark1/checkpoint/full_model.h5')

# Evaluate the model
test_loss, test_accuracy = model.evaluate(test_dataset)
print("Test Loss:", test_loss)
print("Test Accuracy:", test_accuracy)

# Plotting training & validation accuracy and loss values
def plot_history(history):
    # Plot accuracy
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Train Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy over epochs')

    # Plot loss
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Train Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss over epochs')

    plt.show()

# Call the function to plot the graphs
plot_history(history)
