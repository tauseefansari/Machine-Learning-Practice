import cv2
import numpy as np
from os import listdir
from os.path import isfile,join

#Path to file
path="./Faces/Tauseef/"
files=[f for f in listdir(path) if isfile(join(path,f))]

trainingData, Labels = [],[]

for i,file in enumerate(files):
    imagePath=path+files[i]
    images=cv2.imread(imagePath,cv2.IMREAD_GRAYSCALE)
    trainingData.append(np.asarray(images,dtype=np.uint8))
    Labels.append(i)

Labels=np.asarray(Labels,dtype=np.int32)

model = cv2.face.LBPHFaceRecognizer_create()

model.train(trainingData,np.array(Labels))
model.write('trainingData.yml')

def predict_image(test_image):
    img = test_image.copy()
    face, bounding_box = face_detection(img)
    label = model.predict(face)
    label_text = "Tauseef"
    print (label)
    print (label_text)
    (x,y,w,h) = bounding_box
    cv2.rectangle(img, (x,y), (x+w, y+h), (0,255,0), 2)
    cv2.putText(img, label_text, (x,y), cv2.FONT_HERSHEY_PLAIN, 1.5, (0, 255, 0), 2)
    return img


print("Trained")

face_classifier = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

def face_detection(image):
    image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #haar_classifier = cv2.CascadeClassifier('opencv/data/haarcascades/haarcascade_frontalface_default.xml')
    face = face_classifier.detectMultiScale(image_gray, scaleFactor=1.3, minNeighbors=7)
    (x,y,w,h) = face[0]
    return image_gray[y:y+w, x:x+h], face[0]


def FaceDetect(image,size=0.5):
    gray=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=face_classifier.detectMultiScale(gray,1.3,5)
    if faces is ():
        return  image,[]

    for (x,y,w,h) in faces:
        cv2.rectangle(image,(x,y),(x+w,y+h),(0,255,0),5)
        roi=image[y:y+h,x:x+w]
        roi=cv2.resize(roi,(200,200))
        return image,roi

cap=cv2.VideoCapture(0)

while True:
    ret,frame=cap.read()
    image,face=FaceDetect(frame)

    try:
        face=cv2.cvtColor(face,cv2.COLOR_BGR2GRAY)
        fac= model.predict(face)

            #model.predict(face)

        #if result[1] <=500:
            #accuracy=int(100*(1-result[1]/300))
        cv2.putText(fac,"Tauseef",(50,50),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(255,120,150),thickness=5)
        #cv2.putText(image,"Tauseef",cv2.FONT_HERSHEY_COMPLEX,1,(255,120,150),5)
    except:
        cv2.putText(fac,"No Face Found !...",(50,50),fontFace=cv2.FONT_HERSHEY_COMPLEX,fontScale=1,color=(255,120,150),thickness=5)
        cv2.imshow("Face Recognition RealTime",image)
        pass
    if cv2.waitKey(1)==13:
        break
cap.release()
cv2.destroyAllWindows()