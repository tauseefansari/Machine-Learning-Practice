import cv2
import pickle

face_cascade=cv2.CascadeClassifier("Cascade/haarcascade_frontalface_alt.xml")
recognizer=cv2.face.LBPHFaceRecognizer_create()
recognizer.read("trainer.yml")

labels={}
with open("labels.pickle","rb") as f:
    og_labels=pickle.load(f)
    labels={v:k for k,v in og_labels.items()}

frame=cv2.imread("IMG-20190322-WA0147.jpg")
#frame=cv2.resize(frame,(600,600))
gray=cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)
faces=face_cascade.detectMultiScale(gray,scaleFactor=1.5,minNeighbors=5)
for (x,y,w,h) in faces:
    roi_gray=gray[y:y+h,x:x+w]
    roi_color = frame[y:y + h, x:x + w]

    id_,conf=recognizer.predict(roi_gray)

    if conf>=45:# and conf<=85:
        #print(id_)
        #print(labels[id_])
        #cv2.rectangle(frame,(x-1,y-1),(x+w,y-40),(0,255,0),cv2.FILLED)
        cv2.putText(frame,str(labels[id_]),(x+4,y-13),cv2.FONT_HERSHEY_SIMPLEX,1,(255,255,255),2)

    #img_item="my_image.png"
    #cv2.imwrite(img_item,roi_gray)
    #cv2.namedWindow("Face Recognition", cv2.WINDOW_NORMAL)
    #cv2.resize(frame, (600, 600))
    cv2.rectangle(frame,(x,y),(x+w,y+h),(0,255,0),3)
    save_path="./Static/File.jpg"
    cv2.imwrite(save_path,frame)
    #cv2.imshow("Face Recognition",frame)
    #if cv2.waitKey(20) & 0xFF ==ord("q"):
    #   break

#cap.release()
cv2.destroyAllWindows()
