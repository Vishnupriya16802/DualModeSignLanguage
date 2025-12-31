import cv2
import numpy as np
import tensorflow as tf

IMG_SIZE = 128

model = tf.keras.models.load_model("isl_model.h5")

with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img, verbose=0)
    class_id = np.argmax(prediction)
    sign = labels[class_id]

    cv2.putText(frame, f"Sign: {sign}", (40, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

    cv2.imshow("ISL Sign Language Translator", frame)

    if cv2.waitKey(1) & 0xFF == 27:  # ESC key
        break

cap.release()
cv2.destroyAllWindows()

