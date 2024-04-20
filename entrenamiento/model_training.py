import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import Conv2D, MaxPooling2D
from keras.utils import to_categorical
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import seaborn as sns
import tensorflow as tf



classes = ["fernanda", "desconocido"]
img_rows, img_cols = 64, 64 
classes_num = len(classes)


def load_data(): 
    data = [] 
    target = [] 

    for i, label in enumerate(classes): 
        folder_path = os.path.join("entrenamiento/data/", label) 
        for img in os.listdir(folder_path): 
            img_path = os.path.join(folder_path, img)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE) 
            img = cv2.resize(img, (img_rows, img_cols)) 
            data.append(np.array(img)) 
            target.append(i)
    data = np.array(data)  
    data = data.reshape(data.shape[0], img_rows, img_cols, 1) 
    target = np.array(target)
    new_target = to_categorical(target, classes_num)
    return data, new_target 


data, target = load_data()  
x_train, x_test, y_train, y_test = train_test_split( 
    data, target, test_size=0.20, random_state=0
)

model = Sequential()

model.add( Conv2D(32, kernel_size=(3, 3), activation="relu", input_shape=(img_rows, img_cols, 1)))
model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(classes_num, activation="softmax")) 

model.compile(loss="categorical_crossentropy", optimizer="adam", metrics=["accuracy"]) 
history = model.fit(
    x_train,
    y_train,
    batch_size=32,
    epochs=20,
    verbose=1,
    validation_data=(x_test, y_test),
) 

model.save("model.h5") 

saved_model_dir = ""
tf.saved_model.save(model, saved_model_dir)

converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_dir)
tflite_model = converter.convert()

with open("model.tflite", "wb") as f:
    f.write(tflite_model)

if not os.path.exists("training_model/graphs"):
    os.makedirs("training_model/graphs")

y_pred = model.predict(x_test) #predicciones en un conjunto de prueb, usando el modelo entrenado
y_pred_classes = np.argmax(y_pred, axis=1) #Obtiene las clases predichas para cada muestra en el conjunto de prueba
y_true = np.argmax(y_test, axis=1) #Obtiene las clases reales para cada muestra en el conjunto de prueba.

confusion_mtx = confusion_matrix(y_true, y_pred_classes) #Calcula la matriz de confusión utilizando las clases reales y predichas

plt.figure(figsize=(8, 6)) #Crea una nueva figura para el gráfico de la matriz de confusión.
sns.heatmap( #Genera y muestra un gráfico de mapa de calor de la matriz de confusión utilizando la biblioteca Seaborn.
    confusion_mtx,
    annot=True,
    fmt="g",
    cmap="Blues",
    xticklabels=classes,
    yticklabels=classes,
)

plt.xlabel("Prediction") #Genera y muestra un gráfico de la historia del error durante el entrenamiento del modelo. Este gráfico muestra cómo cambia el error (pérdida) en el conjunto de entrenamiento y en el conjunto de prueba a lo largo de las épocas de entrenamiento.
plt.ylabel("Real")
plt.savefig("training_model/graphs/confusion_matrix.png")
plt.show()

plt.figure()
plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Error history")
plt.ylabel("Error")
plt.xlabel("Epoch")
plt.legend(["Train", "Test"], loc="upper right")
plt.savefig("training_model/graphs/error_history.png")
plt.show()