import cv2
org=cv2.imread("check.png")
cv2.imshow("Orignal Image",org)
cv2.waitKey(3000)

#converting to Greyscale
grey=cv2.cvtColor(org,cv2.COLOR_BGR2GRAY)
cv2.imshow("Greyscale Image",grey)
cv2.waitKey(3000)

#writing to File
cv2.imwrite("Greyscale.jpg",grey)

cv2.destroyAllWindows()