import cv2
import numpy as np
#load an Image
image=cv2.imread("Check.png")
image=cv2.resize(image,(1000,700))
#cv2.imshow("Orignal Image",image)
cv2.waitKey(1000)

#Converting to Grayscale because contours work only in Grayscale images in OpenCV
gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)

#finding Canny Edges
edges=cv2.Canny(gray,30,200)
#cv2.imshow("Canny Edges",edges)

#Finding contours
contours,hierarchy=cv2.findContours(edges.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_NONE)

#Drawing Contours
black_image=np.zeros((image.shape[0],image.shape[1],3))
cv2.drawContours(black_image,contours,-1,(0,255,0),2) # -1 for all contours
cv2.imshow("Black Background",black_image)
cv2.drawContours(image,contours,-1,(0,255,0),2) # -1 for all contours

cv2.imshow("Contours",image)
cv2.waitKey()
cv2.destroyAllWindows()