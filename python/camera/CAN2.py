import cv2
import numpy as np
import keras
import serial

# Define the function to process the video stream
def process_frame(frame):
    model = keras.models.load_model('C:/Users/AADIL/Documents/models/model5.h5')
    pic = cv2.resize(frame, (30, 60))
    pic = np.expand_dims(pic, axis=0)
    prediction = model.predict(pic)
    return np.argmax(prediction)

# Communicate with Arduino
ser = serial.Serial('COM3', 9600,)
# Open the video capture device
cap = cv2.VideoCapture(0)


while True:
    if ser.in_waiting > 0:
        response = ser.readline().decode().strip()
        print("Arduino response: "), response
    # Otherwise, run the prediction again on the current frame
        ret, frame = cap.read()
        prediction = process_frame(frame)
        print(prediction)

        if prediction == 0:
            print("sending 'u' to Arduino")
            ser.write(b'u')
        else:
            print("sending 'd' to Arduino")
            ser.write(b'd')
        cv2.imshow('frame', frame)
        cv2.waitKey(0)

    # Release the video capture device and close all windows
        # cap.release()
        cv2.destroyAllWindows()
