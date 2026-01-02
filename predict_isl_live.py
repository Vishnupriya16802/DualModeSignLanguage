import cv2
import numpy as np
import tensorflow as tf
import pyttsx3
import time

# =========================
# Load trained model
# =========================
model = tf.keras.models.load_model("isl_model.h5")

# Load class labels
with open("labels.txt", "r") as f:
    labels = f.read().splitlines()

# =========================
# Text-to-Speech setup
# =========================
engine = pyttsx3.init()
engine.setProperty("rate", 150)

# =========================
# Webcam setup
# =========================
cap = cv2.VideoCapture(0)

# =========================
# Variables for features
# =========================
last_spoken = ""
last_time = 0
SPEAK_DELAY = 2  # seconds

confidence_sum = 0
frame_count = 0
start_time = time.time()

print("🔵 Dual Mode ISL Translator with Evaluation Metrics")

# =========================
# Main loop
# =========================
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # =========================
    # Preprocess frame
    # MUST MATCH TRAINING SIZE
    # =========================
    img = cv2.resize(frame, (128, 128))   # <-- VERY IMPORTANT
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    # =========================
    # Prediction
    # =========================
    preds = model.predict(img, verbose=0)[0]
    index = np.argmax(preds)
    confidence = preds[index] * 100
    label = labels[index]

    # =========================
    # Metrics calculation
    # =========================
    confidence_sum += confidence
    frame_count += 1
    avg_confidence = confidence_sum / frame_count

    elapsed_time = time.time() - start_time
    fps = frame_count / elapsed_time if elapsed_time > 0 else 0

    # =========================
    # Display output
    # =========================
    cv2.putText(frame, f"Sign: {label}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Confidence: {confidence:.2f}%", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

    cv2.putText(frame, f"Avg Confidence: {avg_confidence:.2f}%", (20, 120),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    cv2.putText(frame, f"FPS: {fps:.2f}", (20, 160),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # =========================
    # Auto Speech (Feature 1)
    # =========================
    current_time = time.time()
    if confidence > 80:
        if label != last_spoken and (current_time - last_time) > SPEAK_DELAY:
            engine.say(label)
            engine.runAndWait()
            last_spoken = label
            last_time = current_time

    # =========================
    # Show window
    # =========================
    cv2.imshow("Dual Mode ISL Translator", frame)

    # Exit on ESC
    if cv2.waitKey(1) & 0xFF == 27:
        break

# =========================
# Cleanup
# =========================
cap.release()
cv2.destroyAllWindows()
