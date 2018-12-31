import cv2
import sys

image_path = sys.argv[1]
cascade_path = sys.argv[2]

face_cascade = cv2.CascadeClassifier(cascade_path)

image = cv2.imread(image_path)
gray = cv2.cvtColor(image , cv2.COLOR_BGR2GRAY)

faces = face_cascade.detectMultiScale(
	gray,
	scaleFactor = 1.1,
	minNeighbors = 6,
	minSize = (30,30),
	flags = cv2.CASCADE_SCALE_IMAGE
	)

print('found faces!!')
for x,y,w,h in faces:
	cv2.rectangle(image , (x,y) , (x+w , y+h) , (0,255,0) , 2)

cv2.imshow('faces detected !' , image)
cv2.waitKey(0)


