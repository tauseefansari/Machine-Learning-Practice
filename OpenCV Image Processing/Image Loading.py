import numpy as np
import cv2
#for Loading an Image
image=cv2.imread('Wallpaper-HD.jpg')
cv2.imshow('BWW',image)
height=int(image.shape[0]) #Height of Image
width=int(image.shape[1]) #width of Image
print("Height : ",height,"px","\n","Width : ",width,"px")
#Writing an Image
cv2.imwrite("Check.png",image)
cv2.waitKey(0)
cv2.destroyAllWindows()
