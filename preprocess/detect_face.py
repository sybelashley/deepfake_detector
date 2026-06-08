import cv2
import os

image_path = "dataset/test.jpg"

faces_path = "dataset/faces"

os.makedirs(faces_path, exist_ok=True)

image = cv2.imread(image_path)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

face_cascade = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for i, (x, y, w, h) in enumerate(faces):

    face = image[y:y+h, x:x+w]

    face_path = os.path.join(faces_path, f"face_{i}.jpg")

    cv2.imwrite(face_path, face)

print("Visage détecté avec succès !")