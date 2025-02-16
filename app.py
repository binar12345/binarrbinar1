from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from deepface import DeepFace
import time
import pyttsx3

app = Flask(__name__)

# Load YOLOv8-X (Extra Large) for highest accuracy
model = YOLO("yolov8x.pt")

# Initialize Text-to-Speech (TTS)
tts = pyttsx3.init()
tts.setProperty('rate', 150)

def detect_objects(frame):
    results = model(frame)
    detected_objects = []
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            label = result.names[int(box.cls[0])]
            detected_objects.append(label)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
    return frame, detected_objects

def detect_emotions(frame):
    detected_emotion = ""
    try:
        analysis = DeepFace.analyze(frame, actions=["emotion"], enforce_detection=False)
        detected_emotion = analysis[0]['dominant_emotion']
        cv2.putText(frame, detected_emotion, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)
    except:
        pass
    return frame, detected_emotion

def generate_frames():
    cap = cv2.VideoCapture(0)
    prev_time = time.time()
    
    while True:
        success, frame = cap.read()
        if not success:
            break
        
        frame, detected_objects = detect_objects(frame)
        frame, detected_emotion = detect_emotions(frame)
        
        # Calculate FPS
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time)
        prev_time = curr_time
        cv2.putText(frame, f"FPS: {fps:.2f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 2)
        
        # Speak detected objects & emotions
        if detected_objects or detected_emotion:
            speech_text = ", ".join(detected_objects) + (" with " + detected_emotion if detected_emotion else "")
            tts.say(speech_text)
            tts.runAndWait()
        
        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()
        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")
    cap.release()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/api/detected')
def detected():
    return jsonify({"message": "Live detection is running"})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)
