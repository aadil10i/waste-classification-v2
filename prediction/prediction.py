import matplotlib.pyplot as plt
from keras.models import load_model
import cv2
import numpy as np

IMG_BREDTH = 30
IMG_HEIGHT = 60

# Function to predict waste category using trained model
def use_model(path):
    model = load_model('./cnn_training/models/main_model2.h5')
    pic = plt.imread(path)
    pic = cv2.resize(pic, (IMG_BREDTH, IMG_HEIGHT))
    pic = np.expand_dims(pic, axis=0)
    prediction = model.predict(pic)
    predicted_class = np.argmax(prediction, axis=1)[0]  # Use argmax to find the highest prediction score
    classes = ['Aluminium', 'Organic', 'Paper', 'Plastic']  # Update class names
    return classes[predicted_class]

# Example usage of the model for prediction

print(use_model('C:/Users/AADIL/OneDrive/Documents/WasteImagesDataset/TEST/Plastic/plastic425.jpg'))