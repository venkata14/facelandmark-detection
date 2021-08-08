import numpy as np
import argparse
import cv2



ap = argparse.ArgumentParser()
ap.add_argument("-i", "--image", required=True, help="Path to the input image")
args = vars(ap.parse_args())

face_cascade = cv2.CascadeClassifier("./haarcascade_frontalface_default.xml")
img = cv2.imread(args["image"])
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
faces = face_cascade.detectMultiScale(gray, 1.1, 4)

for (x,y,w,h) in faces:
    cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)

print(faces)
# Show the final result
cv2.imshow("Output", img)
cv2.waitKey(0)