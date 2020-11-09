from PIL import Image
import cv2
from keras.models import load_model
import numpy as np


model = load_model('model.h5')

face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')


def face_extractor(img):
    faces = face_cascade.detectMultiScale(img, 1.3, 3)

    if faces is ():
        return None

    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cropped_face = img[y:y + h, x:x + w]

    return cropped_face


video_capture = cv2.VideoCapture(0)
while True:
    _, frame = video_capture.read()

    face = face_extractor(frame)
    if type(face) is np.ndarray:
        face = cv2.resize(face, (224, 224))
        img = Image.fromarray(face, 'RGB')
        img_array = np.array(img)
        img_array = np.expand_dims(img_array, axis=0)
        pred = model.predict(img_array)
        print(pred)

        name = "Not me"

        if pred[0][1] > 0.5:
            name = 'Me'
        cv2.putText(frame, name, (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    else:
        cv2.putText(frame, "No face found", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
    cv2.imshow('Video', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
cv2.destroyAllWindows()