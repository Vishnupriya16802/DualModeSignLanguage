# Dual Mode Sign Language Translator

This project implements a **dual-mode sign language translation system** using:
- Static image-based Indian Sign Language (ISL)
- Dynamic video-based sign language using WLASL

The system supports real-time prediction and text-to-speech output.

---

## 📂 Dataset Information

Due to large size, datasets are **not included** in this repository.

Please download them from the official sources below:

### 🔹 Indian Sign Language (ISL) Dataset
- Kaggle:  
  https://www.kaggle.com/datasets/prathumarikeri/indian-sign-language-isl

### 🔹 WLASL Dataset
- Official GitHub:  
  https://github.com/dxli94/WLASL
- Dataset page:  
  https://dxli94.github.io/WLASL/

---

## 📌 Project Structure
DualModeSignLanguage/
│── Indian/ # ISL images (download separately)
│── wlasl/ # WLASL dataset (download separately)
│── train_isl_model.py
│── predict_isl_live.py
│── text_to_speech.py
│── labels.txt
│── README.md


---

## 🛠️ Technologies Used
- Python
- OpenCV
- MediaPipe
- TensorFlow / Keras
- NumPy
- Text-to-Speech (pyttsx3)

---

## 📌 Note
Datasets and trained models are excluded from GitHub due to size limitations.
