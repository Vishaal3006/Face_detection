import cv2 as cv
import os
import numpy as np

# List of people to train on
people = ["Einstein", "Will Smith", "Johnny Depp"]

# Directory containing training images
Dir = r"C:\Users\Vishaal\OneDrive\Desktop\Train_Images"

# Load Haar cascade for face detection
haarCascade = cv.CascadeClassifier(cv.data.haarcascades + "haarcascade_frontalface_default.xml")

features = []
labels = []


def face_detect():
    for person in people:
        path = os.path.join(Dir, person)
        label = people.index(person)
        print(f"Processing images for {person} from {path}")
        for img_name in os.listdir(path):
            img_path = os.path.join(path, img_name)
            img_array = cv.imread(img_path)
            if img_array is None:
                print(f"Image {img_path} could not be loaded.")
                continue

            gray = cv.cvtColor(img_array, cv.COLOR_BGR2GRAY)

            faces_rect = haarCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=3)

            for (x, y, w, h) in faces_rect:
                face_roi = gray[y:y + h, x:x + w]

                features.append(face_roi)
                labels.append(label)


face_detect()

# Convert lists to numpy arrays
features = np.array(features, dtype="object")
labels = np.array(labels)

print("__________Training Done_______")

# Initialize the LBPH face recognizer
recognizer = cv.face.LBPHFaceRecognizer_create()

# Train the recognizer with the features and labels
recognizer.train(features, labels)

recognizer.save("face_trained.yml")
# Save the features and labels
np.save("features.npy", features)
np.save("labels.npy", labels)
