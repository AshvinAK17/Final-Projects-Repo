# Emotion Detection from Facial Images

## Objective
Develop a Streamlit web application that lets users upload facial images and automatically predicts the person's emotion (e.g., happy, sad, angry) using a trained Convolutional Neural Network (CNN) 
combined with facial landmarks.

This helps researchers, developers, and hobbyists explore affective computing, understand facial expressions, and build emotion-aware applications.

---

## Project Scope
**End-to-end ML system:**
- Data preprocessing and augmentation
- CNN & landmark MLP model
- Training & evaluation
- Streamlit web app for real-time predictions

**Real-world relevance:**
- Applies computer vision to human emotion recognition
- Interactive tool to visualize and test predictions

---

## Key Components

### 1. User Interface
- Streamlit app that allows image upload (`jpg`, `jpeg`, `png`)
- Displays the uploaded image and:
  - Predicted emotion name
  - Per-class confidence scores

### 2. Image & Landmark Preprocessing
- Detect faces using MTCNN
- Extract 468-point facial landmarks (MediaPipe Face Mesh)
- Resize faces to (128, 128) and normalize pixel values
- Compute mean & std of landmarks for standardization
- Data augmentation: crop, flip, rotation, jitter, random erasing

### 3. Emotion Classification
- Custom CNN extracts image features
- Landmark MLP processes landmark vectors
- Final classifier combines both
- Dataset used: **FER2013 Dataset**  
  [Kaggle - FER2013 Dataset Images](https://www.kaggle.com/datasets/damnithurts/fer2013-dataset-images)

### 4. Performance & Optimization
- Evaluated using accuracy, precision, recall, F1-score
- WeightedRandomSampler to address class imbalance
- Cosine annealing learning rate scheduler
- Early stopping to prevent overfitting
- Test-Time Augmentation (TTA) with horizontal flips

### 5. Deployment & Testing
- Deployed as a Streamlit app
- Tested on various face images

---

## Directory Structure
```
Emotion Detection/
├── ed_final_one_best_model.pth         # Trained model 
├── ed_final_one_train_metrics.json     # Training set evaluation metrics
├── ed_final_one_test_metrics.json      # Test set evaluation metrics
├── Emotion_detection_with_face.ipynb   # Jupyter notebook for Source code
├── ED_streamlit.py                     # Streamlit web app script
├── lm_std.pt                           # Landmark standard deviation tensor
├── lm_mean.pt                          # Landmark mean tensor
├── classes.json                        # List of emotion class labels
└── Emotion_Detection_Report.pdf        # Project report / documentation
```
---

## Tools & Technologies
- Python
- PyTorch
- Streamlit
- facenet-pytorch (MTCNN)
- mediapipe (Face Mesh)
- scikit-learn
- NumPy, tqdm, Pillow

---

## Model Training
The `Emotion_detection_with_face.ipynb` script handles the complete training pipeline:
- Detects faces and extracts landmarks
- Builds a CNN + landmark MLP combined model
- Uses data augmentation for better generalization
- Evaluates using accuracy, precision, recall, F1-score
- Saves:
  - `ed_final_one_best_model.pth` (best model)
  - `lm_mean.pt` and `lm_std.pt` (for landmark standardization)
  - `classes.json` (emotion labels)

> Artifacts help reproduce and deploy the model.

---

## Clone this Repository
1. To get a local copy of this project, run the following command in your terminal:
    
        git clone https://github.com/AshvinAK17/Final-Projects-Repo.git
        cd Final-Projects-Repo/Emotion Detection

2. Install Requirements
   
        pip install -r requirements.txt

3. Run the streamlit app:

       streamlit run ED_streamlit.py

## Streamlit:

You can also directly access the app through the below link: [(https://final-projects-repo-knfbgsjlwooguofn8sabpd.streamlit.app/)](https://final-projects-repo-knfbgsjlwooguofn8sabpd.streamlit.app/)

### Usage:

1.Upload a facial image (.jpg, .jpeg, .png, ≤5MB)

2. The app predicts:
  -Emotion name (e.g., happy, sad)
  -Confidence scores per class

*Note: This application is built for educational & project purposes.
Accuracy depends on data quality and model capacity.*
