import os
#from PIL import Image
from tkinter import messagebox
import tkinter as tk
import numpy as np
import cv2
import pickle


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
image_dir= os.path.join(BASE_DIR,"Images")

face_cascade=cv2.CascadeClassifier("Cascade/haarcascade_frontalface_alt.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()


current_id=0
label_ids={}
x_train=[]
y_labels=[]

for root,dirs,files in os.walk(image_dir):
    for file in files:
        if file.endswith("png") or file.endswith("jpg"):
            path=os.path.join(root,file)
            label=os.path.basename(os.path.dirname(path)).split()[0].capitalize()
            #print(label,path)
            if not label in label_ids:
                label_ids[label]=current_id
                current_id+=1
            id_=label_ids[label]
            #print(label_ids)
            #x_train.append(path)
            #y_labels.append(label)
            #pil_image=Image.open(path).convert("L") #Convert into GRAYSCALE
            gray_image=cv2.imread(path,cv2.IMREAD_GRAYSCALE)
            image_array=np.array(gray_image, "uint8")
            #print(image_array)
            faces=face_cascade.detectMultiScale(image_array,scaleFactor=1.5,minNeighbors=3)

            for (x,y,w,h) in faces:
                roi=image_array[y:y+h,x:x+w]
                x_train.append(roi)
                y_labels.append(id_)

#print(y_labels)
with open("labels.pickle","wb") as f:
    pickle.dump(label_ids,f)

recognizer.train(x_train,np.array(y_labels))
recognizer.save("trainer.yml")

root=tk.Tk()
root.withdraw()
messagebox.showinfo("Face Training","Face Recognizer successfully trained")