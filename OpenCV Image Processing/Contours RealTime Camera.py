import cv2
import numpy as np
def Contours(img):
    #black=np.zeros((img.shape[0],img.shape[1],3))
    gray=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    cannyEdges=cv2.Canny(gray,30,200)
    contours,hierarchy=cv2.findContours(cannyEdges,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)
    #ret = cv2.drawContours(black, contours, -1, (0, 255, 0), 2)
    ret=cv2.drawContours(img,contours,-1,(0,255,0),2)
    return ret

cap=cv2.VideoCapture(0)
while True:
    ret,liveImg=cap.read()
    cv2.namedWindow("Live Contours", cv2.WINDOW_NORMAL)
    cv2.resize(liveImg, (600, 600))
    cv2.imshow("Live Contours", Contours(liveImg))
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()