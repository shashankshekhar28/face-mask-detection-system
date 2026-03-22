# 😷 Real-Time Face Mask Detection using VGG16

A real-time face mask detection system built using Deep Learning (VGG16), OpenCV, and Streamlit WebRTC. This application detects whether a person is wearing a mask or not through a live webcam feed with optimized performance.

---

## 🚀 Features

* 🔍 Real-time face detection using Haar Cascade
* 🧠 Deep Learning model based on VGG16 (Transfer Learning)
* 🎥 Live webcam streaming using Streamlit WebRTC
* ⚡ Optimized inference (prediction every few frames)
* 🎯 Displays prediction with confidence score
* 🟢 Mask detection (Green box)
* 🔴 No Mask detection (Red box)

---

## 🏗️ Tech Stack

* Python
* TensorFlow / Keras
* OpenCV
* Streamlit
* streamlit-webrtc
* NumPy

---

## 📂 Project Structure

```
├── app.py
├── mask_detector_model.keras
├── haarcascade_frontalface_default.xml
├── mask_detection_system_vgg16.ipynb
├── requirements.txt
├── README.md
```

---

## 🧠 Model Details

* Base Model: VGG16 (Transfer Learning)
* Input Size: 224 x 224
* Output: Binary or Multi-class (Mask / No Mask)
* Preprocessing:

  * Rescaling (/255)
  * Face cropping using Haar Cascade

---

## ⚙️ Installation

### 1. Clone the repository

```
git clone https://github.com/your-username/mask-detection.git
cd mask-detection
```

### 2. Create virtual environment

```
python -m venv venv
```

Activate environment:

Windows:

```
venv\Scripts\activate
```

Mac/Linux:

```
source venv/bin/activate
```

### 3. Install dependencies

```
pip install -r requirements.txt
```

---

## ▶️ Run the Application

```
python -m streamlit run app.py
```

Open in browser:

```
http://localhost:8501
```

---

## 🖥️ How It Works

1. Webcam feed is captured using WebRTC
2. Frames are resized for faster processing
3. Faces are detected using Haar Cascade
4. Each face is resized to 224x224 and normalized
5. Passed into trained VGG16 model
6. Prediction is displayed with bounding box and confidence

---

## 📊 Optimization Techniques Used

* Frame skipping (prediction every 5 frames)
* Image resizing (640x480)
* Async video processing
* Cached model loading

---

## 📸 Output

* 😷 Mask → Green bounding box + confidence
* ❌ No Mask → Red bounding box + confidence

---

## ⚠️ Important Note

Update the model path in `app.py`:

```
load_model("path_to_your_model.keras")
```

---

## 🛠️ Future Improvements

* Mobile deployment (Flutter / React Native)
* Edge deployment (Raspberry Pi)
* YOLO-based face detection
* Multi-person tracking
* Cloud deployment (AWS / GCP)

---

## 🤝 Contributing

Contributions are welcome! Feel free to fork this repository and submit a pull request.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## 👨‍💻 Author

Shashank Shekhar
B.Tech CSE | AI/ML Enthusiast
