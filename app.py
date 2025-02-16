<<<<<<< HEAD
from flask import Flask, render_template, Response
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deepface import DeepFace
import time
import os

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load face cascade for emotion detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define dangerous objects
DANGEROUS_OBJECTS = ["knife", "gun", "fire", "weapon"]  # Add more as needed

# Create folder to save dangerous object images
if not os.path.exists("dangerous_objects"):
    os.makedirs("dangerous_objects")

# Function to detect objects
def detect_objects(frame):
    results = model(frame)
    dangerous_detected = False
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0]

            # Check if the detected object is dangerous
            if label in DANGEROUS_OBJECTS and confidence > 0.5:
                dangerous_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for dangerous objects
                cv2.putText(frame, f"DANGER: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for non-dangerous objects
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, dangerous_detected

# Function to detect emotions
def detect_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        try:
            analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except:
            emotion = "Unknown"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame

# Function to generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video feed")

    prev_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect objects and check for dangerous objects
        frame, dangerous_detected = detect_objects(frame)

        # If dangerous object detected, save the frame
        if dangerous_detected:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join("dangerous_objects", f"danger_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"DANGER: Dangerous object detected! Image saved as {image_path}")

        # Detect emotions
        frame = detect_emotions(frame)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    cap.release()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the application
if __name__ == '__main__':
=======
from flask import Flask, render_template, Response
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deepface import DeepFace
import time
import os

app = Flask(__name__)

# Load YOLOv8 model
model = YOLO("yolov8n.pt")

# Load face cascade for emotion detection
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

# Define dangerous objects
DANGEROUS_OBJECTS = ["knife", "gun", "fire", "weapon"]  # Add more as needed

# Create folder to save dangerous object images
if not os.path.exists("dangerous_objects"):
    os.makedirs("dangerous_objects")

# Function to detect objects
def detect_objects(frame):
    results = model(frame)
    dangerous_detected = False
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            confidence = box.conf[0]

            # Check if the detected object is dangerous
            if label in DANGEROUS_OBJECTS and confidence > 0.5:
                dangerous_detected = True
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)  # Red box for dangerous objects
                cv2.putText(frame, f"DANGER: {label}", (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            else:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Green box for non-dangerous objects
                cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    return frame, dangerous_detected

# Function to detect emotions
def detect_emotions(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        try:
            analysis = DeepFace.analyze(face_roi, actions=["emotion"], enforce_detection=False)
            emotion = analysis[0]['dominant_emotion']
        except:
            emotion = "Unknown"
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    return frame

# Function to generate video frames
def generate_frames():
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise Exception("Could not open video feed")

    prev_time = time.time()
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Detect objects and check for dangerous objects
        frame, dangerous_detected = detect_objects(frame)

        # If dangerous object detected, save the frame
        if dangerous_detected:
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            image_path = os.path.join("dangerous_objects", f"danger_{timestamp}.jpg")
            cv2.imwrite(image_path, frame)
            print(f"DANGER: Dangerous object detected! Image saved as {image_path}")

        # Detect emotions
        frame = detect_emotions(frame)

        # Calculate and display FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)

        # Encode frame as JPEG
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    cap.release()

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# Run the application
if __name__ == '__main__':
>>>>>>> b6ee1b3 (Initial commit)
    app.run(debug=True, host="0.0.0.0", port=5000)