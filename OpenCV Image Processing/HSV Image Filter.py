import cv2
image=cv2.imread("check.png")
cv2.imshow("Orignal Iamge",image)
cv2.waitKey(5000)

#converting to BGR Format
b,g,r= cv2.split(image)
cv2.imshow("Red",r)
cv2.imshow("Green",g)
cv2.imshow("Blue",b)

#Again Merge into Orignal
merge=cv2.merge([b,g,r])
cv2.imshow("Merged",merge)

#Amplified (adding Offset)
merge2=cv2.merge([b,g,r+100])
cv2.imshow("Red Amplified",merge2)

#converting to HSV Value
hsv_image=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
cv2.imshow("HSV Image",hsv_image)
cv2.imshow("HUE Channel",hsv_image[:,:,0])
cv2.imshow("SATURATION Channel",hsv_image[:,:,1])
cv2.imshow("VALUE Channel",hsv_image[:,:,2])
cv2.waitKey()
cv2.destroyAllWindows()