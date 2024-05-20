import cv2
import numpy as np
from keras.models import load_model

model=load_model('./trainingDataTarget/model-018.model')

# Load the Haar Cascade classifier

video=cv2.VideoCapture (0)
faceDetect=cv2.CascadeClassifier('haarcascade_frontalface_default.xml') 

color_dict={0:(0,255,0),1:(0,0,255)}
labels_dict={0:'female',1:'Male'}

import cv2

# Initialize the webcam
video = cv2.VideoCapture(0)

# Load the Haar cascade file for face detection
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

while True:
    ret, frame = video.read()
    if not ret:
        break

    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect faces in the frame
    faces = faceDetect.detectMultiScale(gray, 1.3, 3)

    # Draw rectangles around the detected faces
    for (x, y, w, h) in faces:
        sub_face_image=gray[y:y+h, x:x+w]
        resized=cv2.resize(sub_face_image, (32,32))
        normalized=resized/255.0
        reshaped=np.reshape(normalized, (1,32,32,1))
        result=model.predict(reshaped)
        label=np.argmax(result, axis=1)[0]
        print(label)

        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),2)
        cv2.rectangle(frame,(x,y),(x+w,y+h),color_dict[label],2)
        cv2.rectangle(frame,(x,y-40),(x+w,y),color_dict[label],-1)
        cv2.putText(frame,labels_dict[label],(x,y-10),cv2.FONT_HERSHEY_SIMPLEX,0.8,(255,255,255),2)

    # Display the frame
    cv2.imshow("Frame", frame)

    # Wait for 'q' key to exit
    k = cv2.waitKey(1)
    if k == ord('q'):
        break

# Release the video capture object and close all OpenCV windows
video.release()
cv2.destroyAllWindows()
