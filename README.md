
# Face Mask Detection System

This face mask detection application leverages deep learning to identify whether individuals in real-time video feeds are wearing masks, particularly in the context of COVID-19. Utilizing a combination of a pre-trained face detection model and a custom mask detection model, it processes video frames to analyze faces and predict mask usage. The application displays bounding boxes and confidence labels, enhancing safety measures in public spaces. Designed for ease of use, it can be integrated into various surveillance systems for health monitoring during the pandemic.

## Run Locally

Clone the project

```bash
  git clone https://github.com/chamoddissanayake/Face-Mask-Detection-System.git
```

Install dependencies

```bash
  pip install -r requirements.txt
```

Start the Application

```bash
  python detect_mask_video.py
```


## Tech Stack

**Programming Language:** Python

**Deep Learning Framework:** TensorFlow and Keras for model training and predictions

**Computer Vision Library:** OpenCV for image processing and video stream handling

**Data Handling:** NumPy for numerical operations and data manipulation

**Image Processing:** PIL (Pillow) for loading and manipulating images

**Machine Learning Libraries:** Scikit-learn for model evaluation and metrics

**Data Augmentation:** Keras ImageDataGenerator for augmenting training data

**Video Stream Library:** imutils for easier video stream handling

**Matplotlib:** For plotting training loss and accuracy graphs

**Model Serialization:** HDF5 format for saving the trained model