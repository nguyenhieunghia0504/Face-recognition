import cv2
import numpy as np
import os
from PIL import Image

recofnizer = cv2.face.LBPHFaceRecognizer_create()
path = 'dataSet'

def getImageWithID(path):
    imagePaths = [os.path.join(path, f) for f in os.listdir(path)]
    faces = []
    IDs = []

    for imagePath in imagePaths:

        faceImg = Image.open(imagePath).convert('L')
        faceNP = np.array(faceImg, 'uint8')
        print(faceNP)
        Id = int(imagePath.split('\\')[1].split('.')[1])
        faces.append(faceNP)
        IDs.append(Id)

        cv2.imshow('training', faceNP)
        cv2.waitKey(10)

    return faces, IDs


faces, IDs = getImageWithID(path)
recofnizer.train(faces, np.array(IDs))

if not os.path.exists('Recognizer'):
    os.makedirs('Recognizer')

recofnizer.save('Recognizer/TrainningData.yml')
cv2.destroyAllWindows()