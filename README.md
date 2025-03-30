# Lung Cancer Detection using CNN and SVM

## Overview
This project applies Convolutional Neural Networks (CNN) and Support Vector Machines (SVM) for lung cancer detection using medical imaging data. The model is trained to classify images into cancerous or non-cancerous categories.

---



## Features
- Efficient lung cancer detection using CNN and SVM
- Preprocessed medical images for better accuracy
- Visual representation of predictions
- User-friendly UI built with PyQt

---

## Requirements
Ensure you have the following dependencies installed:

```bash
- Python 3.9+
- TensorFlow 2.12+
- NumPy
- OpenCV
- PyQt5
- scikit-learn
- Matplotlib
```

Install dependencies using:
```bash
pip install -r requirements.txt
```

---

# Complete Installation Guide for Lung Cancer Detection Project

## 1. Clone the Repository
To get the project files on your local machine, you first need to clone the GitHub repository. This can be done using the following command:

```bash
git clone https://github.com/dainguyenk2/lung-cancer-cnn-svm.git
cd lung-cancer-cnn-svm
```

- `git clone <repository-url>`: This command copies the entire repository from GitHub to your local machivenv\Scripts\activate
```
- On **macOS/Linux**:
```bash
source venv/bin/activate
```

- `activate`: This activates the virtual environment, allowing you to install and use dependencies without affecting your global Python environment.

### c. Install Dependencies
After activating the virtual environment, install all the necessary dependencies with:

```bash
pip install -r requirements.txt
```

- `pip install -r requirements.txt`: This command reads the `requirements.txt` file in the project and installs all the required Python libraries listed there.

## 3. Run the Application
Once you have installed the dependencies, you can run the application using the following command:

```bash
python app.py
```

- `python app.py`: This command runs the Python application. The file `app.py` should be the entry point for your project (this can vary depending on the structure of the project).

## Summary
To summarize, here are the steps:
1. Clone the repository.
2. Create and activate a virtual environment.
3. Install the dependencies.
4. Run the application.

These commands help you set up and run the project locally

## Usage
- Launch the UI using the `cancer.ui` file.
- Upload an image (chest X-ray).
- The model will classify it as either **Cancerous** or **Non-Cancerous**.

---

## Results
The Lung Cancer Detection using CNN and SVM project combines Convolutional Neural Networks (CNN) for analyzing medical imaging (such as chest X-rays) and Support Vector Machine (SVM) for classifying based on symptoms provided as input.

CNN-based Diagnosis (Image Analysis):
Convolutional Neural Networks (CNN): CNNs are used for processing chest X-ray images to automatically extract important features such as tumors, nodules, and abnormal areas in the lungs. These features are then used to classify the image as Cancerous or Non-Cancerous.

Image Classification: The CNN model is trained on a large dataset of labeled X-ray images, enabling it to recognize patterns that are associated with lung cancer and classify new images accurately.

SVM-based Diagnosis (Symptom-Based Analysis):
Support Vector Machine (SVM): The SVM classifier is used to analyze the symptoms provided by the patient, such as persistent cough, chest pain, shortness of breath, and other clinical indicators. The model uses this data to classify whether the symptoms are likely to be associated with lung cancer.

Symptom Classification: By training the SVM on a dataset of patient symptoms and outcomes, the model is able to detect patterns and predict whether a patient is likely to have lung cancer based on their reported symptoms.

Overall Results:
Accuracy: The combined model achieves an accuracy of up to 95%, effectively classifying between cancerous and non-cancerous cases based on both images and symptoms.

Precision and Recall: The model strikes a balance between Precision and Recall, minimizing false positives and false negatives in both image-based and symptom-based diagnoses.

These results show that the model, leveraging both CNN for image analysis and SVM for symptom-based classification, can serve as an effective tool for lung cancer detection, helping doctors make more accurate and timely diagnoses.
---

## Technologies Used
- **CNN (Convolutional Neural Network)** for image feature extraction
- **SVM (Support Vector Machine)** for classification
- **TensorFlow** for deep learning
- **OpenCV** for image preprocessing
- **PyQt5** for GUI development

---

🩺 This project is intended for research and educational purposes only. It is not a certified medical application.



