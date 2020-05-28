import cv2
image=cv2.imread("Check.png")
image=cv2.resize(image,(1000,700))
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
ret,thresh=cv2.threshold(gray,127,255,cv2.THRESH_BINARY_INV)
contours,tres=cv2.findContours(thresh.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

for c in contours:
    x,y,w,h=cv2.boundingRect(c)
    cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),3)
    cv2.imshow("Bounding Rectangle",image)

for c in contours:
    accuracy=0.05*cv2.arcLength(c,True)
    approx=cv2.approxPolyDP(c,accuracy,True)
    cv2.drawContours(image,[approx],0,(0,255,0),3)
    cv2.imshow("Approx Poly DP",image)

cv2.waitKey()
cv2.destroyAllWindows()