import tensorflow as tf

# Load the model from the latest checkpoint
model = tf.keras.models.load_model('G:/mark1/checkpoints')

# Save the model
model.save('G:/mark1/model_1')
