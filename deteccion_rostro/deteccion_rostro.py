from keras.models import load_model
import cv2
import numpy as np

model = load_model("model.h5")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

source = cv2.VideoCapture(0)

classes = ["fernanda", "desconocido"]
colors = {
    "fernanda": (255, 0, 100),
    "desconocido": (0, 0, 255),
}

while True:
    ret, img = source.read()
    faces = face_classifier.detectMultiScale(img, 1.3, 5)

    for x, y, w, h in faces:
        face_img = img[y:y+h, x:x+w]  # Extraer la cara de la imagen original
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (64, 64))
        face_img = np.array(face_img).reshape(-1, 64, 64, 1) # Agregar una dimensión extra para el batch

        result = model.predict(face_img)
        predicted_class = classes[np.argmax(result)]

        cv2.putText(
            img,
            "{}".format(predicted_class),
            (x, y - 25),
            2,
            1.1,
            (colors[predicted_class]),
            1,
            cv2.LINE_AA,
        )
        cv2.rectangle(img, (x, y), (x + w, y + h), (colors[predicted_class]), 2)

    cv2.imshow("Live capture", img)

    key = cv2.waitKey(1)
    if key == 27:
        break

source.release()
cv2.destroyAllWindows()
