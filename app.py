from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from keras.models import model_from_json

app = Flask(__name__)

# Load Model
json_file = open("emotiondetector.json", "r")
model_json = json_file.read()
json_file.close()
model = model_from_json(model_json)
model.load_weights("emotiondetector.h5")

# Load OpenCV Face Detector
haar_file = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
face_cascade = cv2.CascadeClassifier(haar_file)

labels = {0: 'angry', 1: 'disgust', 2: 'fear', 3: 'happy', 4: 'neutral', 5: 'sad', 6: 'surprise'}
latest_emotion = "Detecting..."

def extract_features(image):
    feature = np.array(image).reshape(1, 48, 48, 1)
    return feature / 255.0

def detect_emotion():
    global latest_emotion
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (p, q, r, s) in faces:
            image = gray[q:q+s, p:p+r]
            image = cv2.resize(image, (48, 48))
            img = extract_features(image)
            pred = model.predict(img)
            latest_emotion = labels[pred.argmax()]

            # Draw on Frame
            cv2.rectangle(frame, (p, q), (p + r, q + s), (255, 0, 0), 2)
            cv2.putText(frame, latest_emotion, (p, q - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        _, buffer = cv2.imencode(".jpg", frame)
        frame = buffer.tobytes()

        yield (b"--frame\r\n"
               b"Content-Type: image/jpeg\r\n\r\n" + frame + b"\r\n")

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(detect_emotion(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/current_emotion')
def current_emotion():
    return jsonify({"emotion": latest_emotion})

if __name__ == '__main__':
    app.run(debug=True)
