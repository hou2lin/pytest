#!/usr/bin/env python
import cv2
import rospy

#img = cv2.imread("a.jpg")
cv2.namedWindow("Image")
cap = cv2.VideoCapture(0)
while(1):
	ret,img = cap.read()
	cv2.imshow("Image",img)
	if cv2.waitKey(1) & 0xFF == ord('q'):
		break
cap.release()
cv2.destroyAllWindows()
