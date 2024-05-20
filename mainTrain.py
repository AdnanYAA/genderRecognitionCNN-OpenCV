import os
import cv2
import numpy as np
import tensorflow as tf

# Example: Importing to_categorical
# from tensorflow.python.keras.utils import to_categorical
from keras.utils import to_categorical

# Set the correct datapath
datapath = 'Dataset'

# Get the list of classes
classes = os.listdir(datapath)

# Create a dictionary to map class names to labels
label_dict = {class_name: idx for idx, class_name in enumerate(classes)}
print(label_dict)

img_size = 32
data = []
target = []

# Load the Haar Cascade classifier
facedata = "haarcascade_frontalface_default.xml"
cascade = cv2.CascadeClassifier(facedata)

# Iterate over each class
for category in classes:
    folder_path = os.path.join(datapath, category)
    img_names = os.listdir(folder_path)

    for img_name in img_names:
        img_path = os.path.join(folder_path, img_name)
        img = cv2.imread(img_path)
        
        if img is None:
            print(f"Failed to load image: {img_path}")
            continue
        
        faces = cascade.detectMultiScale(img)

        try:
            for x, y, w, h in faces:
                sub_face = img[y:y + h, x:x + w]
                gray = cv2.cvtColor(sub_face, cv2.COLOR_BGR2GRAY)
                resized = cv2.resize(gray, (img_size, img_size))
                data.append(resized)
                target.append(label_dict[category])
        except Exception as e:
            print(f"Error processing image {img_name}: {e}")

# Preprocessing
data = np.array(data) / 255.0
data = np.reshape(data, (data.shape[0], img_size, img_size, 1))
target = np.array(target)

# Convert target to categorical
new_target = tf.keras.utils.to_categorical(target)

# Save the preprocessed data
if not os.path.exists("./trainingDataTarget"):
    os.makedirs("./trainingDataTarget")

np.save("./trainingDataTarget/data", data)
np.save("./trainingDataTarget/target", new_target)
