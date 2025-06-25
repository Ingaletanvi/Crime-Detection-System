<div align="center">

# 🔍 Crime Detection System Using YOLOv8 & 3D CNN  
**Real-Time Video Surveillance & Automated Alerting**

</div>

---

## 📌 Overview

This project integrates object detection (YOLOv8) and action recognition (3D CNN) to detect criminal activity such as the presence of weapons or violent actions in video footage. Built using Streamlit, it offers a user-friendly interface for video upload or live CCTV analysis and sends alerts via email when a crime is detected.

---

## 👤 Team

- **Tanvi Ingale**

---

## 🎯 Objective

To build a hybrid video surveillance system capable of detecting both **objects (weapons)** and **actions (suspicious/criminal behavior)** using deep learning techniques.

---

## 🗃️ Dataset

- 📦 Custom video dataset sourced from **YouTube Surveillance Footage** and **Kaggle Crime Datasets**
    - Classes: `gun`, `knife`, `normal`
    - Total clips: ~20 videos  
    - Extracted frames per clip: 16  
- Object detection annotation dataset (YOLO format)  
- Optional: [SmartCity CCTV Violence Dataset](https://www.kaggle.com/datasets) *(as reference)*

---

## 🧠 Models Used

| Task              | Model      | Framework     |
|-------------------|------------|----------------|
| Object Detection  | YOLOv8     | Ultralytics    |
| Action Recognition| 3D CNN     | Keras / TensorFlow |

---

## 🛠️ Project Steps

1. **Data Collection**: Gathered video samples and categorized them into normal and crime-related videos (knife/gun).
2. **Frame Extraction**: Converted each video into 16 equally spaced frames for CNN processing.
3. **YOLOv8 Training**: Trained a custom YOLOv8 model to detect weapons.
4. **3D CNN Training**: Built and trained a 3D CNN to classify human activity (normal/criminal).
5. **Model Integration**: Combined both models in a Streamlit-based UI.
6. **Live Testing**: Enabled support for live CCTV feeds or uploaded video analysis.
7. **Alert System**: Configured email alerts when criminal activity is detected.
8. **User Authentication**: Added a basic login system to restrict access.
9. **Performance Monitoring**: Integrated accuracy/loss tracking and confusion matrix generation.
10. **Deployment Ready**: Final testing and optimization for user interaction.

---

## 🚀 Features

- Upload or stream live CCTV video
- YOLOv8-based object detection (weapons)
- 3D CNN for activity classification
- Email alert system for real-time notification
- Streamlit login interface for basic access control

---

## 📌 Future Enhancements

- Add facial recognition for criminal identification  
- Store video logs and predictions in a database  
- Deploy on cloud (AWS/GCP) for continuous monitoring  
- Integrate with law enforcement APIs or alert systems  
