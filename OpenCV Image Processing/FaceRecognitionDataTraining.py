import os
from tkinter import messagebox
import cv2
from tkinter import *

name=""

def getName():
    global name
    name=nameInput.get()
    root.destroy()

root=Tk()
root.resizable(0,0)
root.title("User Information")
lbl=Label(root,text="Enter Your Name: ",font=("Arial", 20,"italic")).pack()
nameInput=Entry(root, font=("Arial", 20,"italic"))
nameInput.pack()
btnSubmit=Button(root,text="Save",font=("Arial", 20,"italic"),command=getName)
btnSubmit.pack()
root.mainloop()

#Detect Face using Haar Cascade Classifier
face_classifier=cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def DetectFace(image):
    grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(grayscale,1.3,5)
    for (x,y,w,h) in faces:
        image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
    return image

def CropFace(image):
    grayscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(grayscale,1.3,5)

    if faces is ():
        return None

    for (x,y,w,h) in faces:
        cropFace=image[y:y+h,x:x+w]

    return cropFace

cap=cv2.VideoCapture(0)
NoOfImageTrained=0
MaxLimit=50

#Create Directory
os.chdir(os.getcwd()+"/Faces")
try:
    os.mkdir(name.capitalize())
except:
    messagebox.showinfo("User Information","Already Created! Replacing with the new images!")
os.chdir(name)


while True:
    ret,liveFrame=cap.read()
    if CropFace(liveFrame) is not None:
        NoOfImageTrained+=1
        faceFromImage=cv2.resize(CropFace(liveFrame),(200,200))
        faceFromImage=cv2.cvtColor(faceFromImage,cv2.COLOR_BGR2GRAY)

        #save path
        savePath=str(NoOfImageTrained)+".jpg"
        cv2.imwrite(savePath,faceFromImage)

        cv2.putText(faceFromImage,str(NoOfImageTrained),(50,50),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),3)
        cv2.imshow("Face Extractor",DetectFace(liveFrame))

    else:
        pass

    if cv2.waitKey(1)==13 or NoOfImageTrained==50:    #13 for Enter
        break

cap.release()
cv2.destroyAllWindows()



