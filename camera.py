import cv2
import numpy as np
from keras.models import load_model

# Load the trained model
model = load_model('./trainingDataTarget/model-018.model')

# Load the Haar Cascade classifier for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

# Dictionary for color and labels
color_dict = {0: (0, 255, 0), 1: (0, 0, 255)}
labels_dict = {0: 'female', 1: 'Male'}

class Video(object):
    def __init__(self):
        self.video = cv2.VideoCapture(0)

    def __del__(self):
        if self.video.isOpened():
            self.video.release()

    def get_frame(self):
        ret, frame = self.video.read()
        if not ret:
            return None

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = faceDetect.detectMultiScale(gray, 1.3, 5)
        for (x, y, w, h) in faces:
            x1, y1 = x + w, y + h
            cv2.rectangle(frame, (x, y), (x1, y1), (255, 0, 255), 1)
            cv2.line(frame, (x, y), (x + 30, y), (255, 0, 255), 6)  # Top Left
            cv2.line(frame, (x, y), (x, y + 30), (255, 0, 255), 6)
            cv2.line(frame, (x1, y), (x1 - 30, y), (255, 0, 255), 6)  # Top Right
            cv2.line(frame, (x1, y), (x1, y + 30), (255, 0, 255), 6)
            cv2.line(frame, (x, y1), (x + 30, y1), (255, 0, 255), 6)  # Bottom Left
            cv2.line(frame, (x, y1), (x, y1 - 30), (255, 0, 255), 6)
            cv2.line(frame, (x1, y1), (x1 - 30, y1), (255, 0, 255), 6)  # Bottom Right
            cv2.line(frame, (x1, y1), (x1, y1 - 30), (255, 0, 255), 6)

            sub_face_image = gray[y:y + h, x:x + w]
            resized = cv2.resize(sub_face_image, (32, 32))
            normalized = resized / 255.0
            reshaped = np.reshape(normalized, (1, 32, 32, 1))
            result = model.predict(reshaped)
            label = np.argmax(result, axis=1)[0]
            print(label)

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            # cv2.rectangle(frame, (x, y), (x + w, y + h), color_dict[label], 2)
            cv2.rectangle(frame, (x, y - 40), (x + w, y), color_dict[label], -1)
            cv2.putText(frame, labels_dict[label], (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

        ret, jpeg = cv2.imencode('.jpg', frame)
        return jpeg.tobytes()


