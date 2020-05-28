import numpy as np
import cv2
def ShowSketch(image):
    grayScale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    imgGrayBlur=cv2.GaussianBlur(grayScale,(5,5),0)
    cannyEdge=cv2.Canny(imgGrayBlur,5,50)
    ret,maskImage=cv2.threshold(cannyEdge,100,255,cv2.THRESH_BINARY_INV)
    return maskImage

cap=cv2.VideoCapture(0)
while True:
    ret,liveImg=cap.read()
    cv2.namedWindow("Live Sketch",cv2.WINDOW_NORMAL)
    cv2.resize(liveImg,(600,600))
    cv2.imshow("Live Sketch", ShowSketch(liveImg))
    if cv2.waitKey(1) == 13: # for Enter key code is 13
        break
cap.release()
cv2.destroyAllWindows()