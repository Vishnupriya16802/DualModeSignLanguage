import tensorflow as tf
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.metrics import accuracy_score, classification_report

# =============================
# Paths
# =============================
MODEL_PATH = "isl_model.h5"
DATASET_PATH = "Indian"   # <-- your dataset folder
IMG_SIZE = (128, 128)
BATCH_SIZE = 32

# =============================
# Load model
# =============================
model = tf.keras.models.load_model(MODEL_PATH)

# =============================
# Load validation data
# =============================
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

val_generator = datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

class_names = list(val_generator.class_indices.keys())

# =============================
# Predictions
# =============================
y_true = val_generator.classes
y_pred_probs = model.predict(val_generator)
y_pred = np.argmax(y_pred_probs, axis=1)

# =============================
# Evaluation Metrics
# =============================
accuracy = accuracy_score(y_true, y_pred)

print("\n==============================")
print(f"✅ FINAL VALIDATION ACCURACY: {accuracy * 100:.2f}%")
print("==============================\n")

print("📊 CLASSIFICATION REPORT:")
print(classification_report(y_true, y_pred, target_names=class_names))
