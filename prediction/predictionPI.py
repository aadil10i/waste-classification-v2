from turtle import delay
from gpiozero import Motor, Servo
import cv2
import numpy as np
from tensorflow.keras.models import load_model  # TensorFlow backend

# Import TensorFlow Lite interpreter
import tflite_runtime.interpreter as tflite

# Define GPIO pins
IN1_pin = 23  # DC motor driver pin (B-1A)
IN2_pin = 24  # DC motor driver pin (B-1B)
servo_pin = 25  # Servo motor control pin

# Define motor and servo objects
motor = Motor(IN1_pin, IN2_pin)
servo = Servo(servo_pin)

# Define duty cycles for different compartments
duty_cycle_paper = 20  # Adjust these values as needed
duty_cycle_plastic = 50
duty_cycle_organic = 70

# Load the pre-trained machine learning model
# model = load_model('path/to/your/model.h5')

# Load the TensorFlow Lite model
interpreter = tflite.Interpreter(model_path='path/to/your/model.tflite')
interpreter.allocate_tensors()

# Define waste categories based on prediction output
categories = {0: "Organic", 1: "Paper", 2: "Plastic", 3: "Aluminium"}

# Function to pre-process the image for prediction
def pre_process_image(frame):
  resized_image = cv2.resize(frame, (30, 60))  # Resize to model input size
  return resized_image.astype('float32') / 255.0  # Normalize pixel values

# Function to identify waste type using the model 
# def predict_waste(frame):
#   preprocessed_image = pre_process_image(frame)
#   image_batch = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension
#   prediction = model.predict(image_batch)
#   predicted_class = np.argmax(prediction)
#   return categories[predicted_class]

# Function to identify waste type using the TensorFlow Lite model
def predict_waste(frame):
  preprocessed_image = pre_process_image(frame)
  image_batch = np.expand_dims(preprocessed_image, axis=0)  # Add batch dimension

  # Get input and output tensors
  input_details = interpreter.get_input_details()
  output_details = interpreter.get_output_details()

  # Set input tensor
  interpreter.set_tensor(input_details[0]['index'], image_batch)

  # Run inference
  interpreter.invoke()

  # Get output tensor
  prediction = interpreter.get_tensor(output_details[0]['index'])

  predicted_class = np.argmax(prediction)
  return categories[predicted_class]

# Open the PiCamera 
cap = cv2.VideoCapture(0)

while True:
  # Capture a frame from the camera
  ret, frame = cap.read()

  # Get waste prediction using the model
  waste_type = predict_waste(frame)
  print(f"Predicted Waste Type: {waste_type}")

  # Move motor to respective compartment based on prediction
  if waste_type == "Organic":
    motor.move(duty_cycle_organic)
  elif waste_type == "Paper":
    motor.move(duty_cycle_paper)
  elif waste_type == "Plastic":
    motor.move(duty_cycle_plastic)
  else:  # Aluminium or unknown category (adjust as needed)
    motor.stop()

  # Open servo flap to drop waste
  servo.angle = 90
  delay(1)  # Adjust delay for servo movement

  # Reset motor and servo positions
  motor.stop()
  servo.angle = 0

  # Display the frame
  cv2.imshow('Waste Sorting System', frame)
  if cv2.waitKey(1) == ord('q'):
    break

# Release resources
cap.release()
cv2.destroyAllWindows()
