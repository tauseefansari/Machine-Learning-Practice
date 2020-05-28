import cv2
import pickle

face_cascade=cv2.CascadeClassifier("Cascade/haarcascade_frontalface_alt.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={}
with open("labels.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

cap=cv2.VideoCapture(0)
while True:
    ret,frame=cap.read()
    gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
    faces=face_cascade.detectMultiScale(gray,scaleFactor=2.0,minNeighbors=4)
    for (x,y,w,h) in faces:
        roi_gray=gray[y:y+h,x:x+w]
        roi_color = frame[y:y + h, x:x + w]

        id_,conf=recognizer.predict(roi_gray)

        if conf>=100:# and conf<=85:
            #print(id_)
            #print(labels[id_])
            #cv2.rectangle(frame,(x-1,y-1),(x+w,y-40),(0,255,0),cv2.FILLED)
            cv2.putText(frame,str(labels[id_]),(x+4,y-13),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)
        else:
            cv2.putText(frame,"Unknown",(x+4,y-13),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

        #img_item="my_image.png"
        #cv2.imwrite(img_item,roi_gray)
        cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
        cv2.resize(frame, (600, 600))
        cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)

    cv2.imshow("Face Recognition",frame)
    if cv2.waitKey(20) & 0xFF ==ord("q"):
        break

cap.release()
cv2.destroyAllWindows()
