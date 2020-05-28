import cv2
from flask import request,Response,Flask
import uuid
import json
import numpy as np
import os.path

def faceDetect(image):
    cascade_face=cv2.CascadeClassifier("Cascade/haarcascade_frontalface_alt.xml")
    garyscale=cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
    faces=cascade_face.detectMultiScale(garyscale,1.3,5)
    for (x,y,w,h) in faces:
        image=cv2.rectangle(image,(x,y),(x+w,y+h),(0,0,255),10)
    pathFile=('static/%s.jpg'%uuid.uuid4().hex)
    cv2.imwrite(pathFile,image)
    return json.dumps(pathFile)

app=Flask(__name__)
@app.route('/api/upload',methods=['POST'])
def upload():
    image=cv2.imdecode(np.frombuffer(request.files['image'].read(),np.uint8),cv2.IMREAD_UNCHANGED)
    processImage=faceDetect(image)
    return Response(response=processImage,status=200,mimetype="application/json")

app.run(host="0.0.0.0",port=5000)

