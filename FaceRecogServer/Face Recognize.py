import cv2
from flask import request,Response,Flask
import uuid
import json
import os
import numpy as np
import pickle

def faceRecog(image):
    #image=cv2.resize(img,(400,400))
    face_cascade = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_alt.xml")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.read("trainer.yml")
    labels = {}
    with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y + h, x:x + w]
        id_, conf = recognizer.predict(roi_gray)
        if conf >= 45:
            cv2.putText(image, str(labels[id_]), (x + 4, y - 13), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    pathFile = ('static/%s.jpg' % uuid.uuid4().hex)
    cv2.imwrite(pathFile, image)
    return json.dumps(pathFile)

def AddCriminal(name,image):
    #image=cv2.resize(img,(400,400))
    #face_cascade = cv2.CascadeClassifier("Cascade/haarcascade_frontalface_alt.xml")
    #recognizer = cv2.face.LBPHFaceRecognizer_create()
    #recognizer.read("trainer.yml")
    labels = {}
    with open("labels.pickle", "rb") as f:
        og_labels = pickle.load(f)
        labels = {v: k for k, v in og_labels.items()}
    #gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    #faces = face_cascade.detectMultiScale(gray, scaleFactor=1.5, minNeighbors=5)
    #for (x, y, w, h) in faces:
    #    roi_gray = gray[y:y + h, x:x + w]
    #    id_, conf = recognizer.predict(roi_gray)
    #    if conf >= 45:
    #        cv2.putText(image, str(labels[id_]), (x + 4, y - 13), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    #    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 3)
    """if name not in labels.values():
        os.mkdir(name)
        os.chdir(name)
    pathFile = ('images/%s.jpg' % uuid.uuid4().hex)
    cv2.imwrite(pathFile, image)
    return json.dumps(pathFile)"""



app=Flask(__name__)
@app.route('/api/upload',methods=['POST'])
def upload():
    image=cv2.imdecode(np.frombuffer(request.files['image'].read(),np.uint8),cv2.IMREAD_UNCHANGED)
    processImage=faceRecog(image)
    return Response(response=processImage,status=200,mimetype="application/json")

app.run(host="0.0.0.0",port=5000)

