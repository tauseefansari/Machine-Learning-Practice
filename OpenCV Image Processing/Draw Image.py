import cv2
import numpy as np
image=np.zeros((512,512,3),np.uint8) #Black Background here uint = Unsigned Integer that takes whole nos.
image.fill(255) #for White Background
#image[:]=255 #or Do this for white background
cv2.imshow("White Image",image)
cv2.waitKey(5000)
cv2.destroyAllWindows()

#Draw Lines
cv2.line(image,(0,0),(511,511),(255,0,255),10) #Line
cv2.line(image,(511,0),(0,511),(255,0,255),10) #Line
cv2.imshow("Line Image",image) #Display Lines
cv2.waitKey(5000)
cv2.destroyAllWindows()

#Draw Rectangle
cv2.rectangle(image,(100,100),(400,400),(255,0,0),5)
cv2.imshow("Rectangle Image",image) #Display Rectangle
cv2.waitKey(5000)
cv2.destroyAllWindows()

#Draw Circle
cv2.circle(image,(250,250),50,(0,255,0),-1)
cv2.imshow("Circle Image",image) #Display Rectangle
cv2.waitKey(5000)
cv2.destroyAllWindows()