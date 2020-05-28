import cv2
import numpy as np
#Translate Image
#                       Translation Matrix: M : | 1    0   Tx  |
#                                              | 0     1   Ty |
image=cv2.imread("Check.png")
height,width=image.shape[:2]
Tx,Ty=height/4,width/4 #Quarter of Height and Width
T=np.float32([[1,0,Tx],[0,1,Ty]])
translated_image=cv2.warpAffine(image,T,(width,height))
cv2.imshow("Translated Image",translated_image)
cv2.waitKey(5000)
cv2.destroyAllWindows()

#Rotation Image
#                       Rotation Matrix: M :    | cosΘ  -sinΘ |
#                                              | sinΘ   cosΘ |
centerx,centery=width/2,height/2
rotationMatrix=cv2.getRotationMatrix2D((width/2,height/2),90,1) #Angle in Anticlockwise
rotatedImage=cv2.warpAffine(image,rotationMatrix,(width,height))
cv2.imshow("Rotated Image",rotatedImage)
cv2.waitKey(5000)
cv2.destroyAllWindows()

#Rotation Image (Uses the concept of interpolation)
resizedImage=cv2.resize(image,(900,500),interpolation=cv2.INTER_AREA)
cv2.imshow("Resized Image",resizedImage)
cv2.waitKey(5000)
cv2.destroyAllWindows()

#image Pyramids (Another way of Resizing)
"""for i in range(0,3):         different small images
    small=cv2.pyrDown(image)
    cv2.imshow("Small Image", small)
    cv2.waitKey(5000)
    image=small"""
small=cv2.pyrDown(image)
large=cv2.pyrUp(small)
cv2.imshow("Small Image",small)
cv2.waitKey(5000)
cv2.destroyAllWindows()
cv2.imshow("Large Image",large)
cv2.waitKey(5000)
cv2.destroyAllWindows()

#image Cropping
startx,starty=int(width*0.25),int(height*0.25) #top left corner
endx,endy=int(width*0.75),int(height*0.75) #bottom right corner
cropped=image[startx:endx,starty:endy]
cv2.imshow("Cropped Image",cropped)
cv2.waitKey(5000)
cv2.destroyAllWindows()

#Brightness and Darkness
brightDark=np.ones(image.shape,dtype="uint8") * 100
bright=cv2.add(image,brightDark)
cv2.imshow("Bright Image",bright)
cv2.waitKey(5000)
cv2.destroyAllWindows()
dark=cv2.subtract(image,brightDark)
cv2.imshow("Dark Image",dark)
cv2.waitKey(5000)
cv2.destroyAllWindows()