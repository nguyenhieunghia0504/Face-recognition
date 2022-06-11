import cv2
import numpy as np
import os
import datetime
import sqlite3
from PIL import Image

#trainning hinh anh nhan dien voi Thu vien nhan dien khuon mat
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
recofnizer = cv2.face.LBPHFaceRecognizer_create()
kiemtra = False
recofnizer.read(r'./Recognizer/TrainningData.yml')
conn = sqlite3.connect('./Recognizer/data_face_recognition.db')
#lay thong tin cua nguoi trong database


def getProfile(id):
    query = "SELECT * FROM people WHERE ID="+str(id)
    cusror = conn.execute(query)

    profile = None
    for row in cusror:
        profile = row
    return profile

datetime_object = datetime.datetime.now()
cap = cv2.VideoCapture(0)
fontface = cv2.FONT_HERSHEY_SIMPLEX
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray)
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 225, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]
        id, confidence = recofnizer.predict(roi_gray)
        if confidence<40:
            profile = getProfile(id)
            if(profile != None):
                cv2.putText(frame, ""+str(profile[1]), (x+10, y+h+ -100), fontface, 1, (0, 255, 0), 2)
                kiemtra = True
        else:
            cv2.putText(frame, "Unknow", (x + 10, y + h + -100), fontface, 1, (0, 0, 255), 2)
    cv2.imshow('image', frame)
    if(cv2.waitKey(1) == ord('q')):
        break

if kiemtra == True:
   conn.execute("INSERT INTO diemdanh(id_user, NgayDiemDanh) VALUES(" + str(profile[0]) + ", '" + str(datetime_object) + "')")
   conn.commit()


cap.release()
cv2.destroyWindow()
conn.close()

