import tensorflow as tf

# Load your Keras model
model = tf.keras.models.load_model('C:/Users/AADIL/MAIN/cnn_training/models/main_model2.h5')

# Create a TFLite converter
converter = tf.lite.TFLiteConverter.from_keras_model(model)

# Convert the model
tflite_model = converter.convert()

# Save the TensorFlow Lite model file
with open('./main_model2.tflite', 'wb') as f:
    f.write(tflite_model)