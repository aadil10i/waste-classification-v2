import numpy as np
import cv2
import tensorflow as tf
import keras
from keras.models import Sequential, load_model
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten, Dropout

import serial

def use_model(frame):
    model = load_model('C:/Users/AADIL/Documents/models/model5.h5')
    pic = cv2.resize(frame, (30, 60))
    pic = np.expand_dims(pic, axis=0)
    classes = model.predict_classes(pic)
    predictions = model.predict(pic)	
    return classes

ser = serial.Serial('COM3', 9600)

while True:
	if(ser.in_waiting > 0):
		line = ser.readline()

		cap = cv2.VideoCapture(0) # video capture source camera (Here webcam of laptop) 
		ret,frame = cap.read() # return a single frame in variable `frame`

		# cv2.imwrite('waste.jpg', frame)

		predict = use_model(frame)[0]

		print(predict)

		if predict == 0:
			ser.write(b'u')
		else:
			ser.write(b'd')

		cv2.imshow('image',frame)
		cv2.waitKey(0)
		cv2.destroyAllWindows()
