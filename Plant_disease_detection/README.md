#### Plant Disease Detection from Images

## Objective
Develop a **Streamlit web application** that lets users upload images of plant leaves and automatically predicts the type of disease (if any) using a trained Convolutional Neural Network (CNN).  
This helps farmers and gardeners quickly identify plant diseases and take corrective action.

---

## Project Scope
 End-to-end ML system:  
- Data preparation and augmentation  
- CNN & transfer learning models  
- Training & evaluation  
- Streamlit-based web app for real-time predictions

 Real-world relevance:  
- Applies computer vision to agriculture  
- Offers an accessible tool for disease diagnosis

---

###  Key Components

## 1. User Interface
- **Streamlit app** that allows image upload (`jpg`, `jpeg`, `png`)
- Displays uploaded image and prediction result (disease name + confidence)

## 2. Image Preprocessing
- Resize images to (128, 128)
- Normalize pixel values
- Data augmentation (rotation, shift, zoom, etc.)

## 3. Disease Classification
- Trained a **Custom CNN**
- Compared performance with **MobileNetV2, VGG16, ResNet50** (transfer learning)
- Dataset: [New Plant Diseases Dataset (Augmented)](https://www.kaggle.com/vipoooool/new-plant-diseases-dataset)

## 4. Performance & Optimization
- Evaluated with accuracy, precision, recall, F1-score
- Used callbacks (early stopping & learning rate reduction)

## 5. Deployment & Testing
- Deployed as a **Streamlit app**
- Tested with various leaf images

---

###  Directory Structure

├── models/ # Saved trained models (.h5)
├── plots/ # Accuracy plots
├── metrics_reports/ # JSON reports with evaluation metrics
├── plant_disease_streamlit.py # Streamlit web app script
├── plant_disease.py # Training and evaluatingscript
├── requirements.txt
└── README.md

---

###  Tools & Technologies
- Python
- TensorFlow & Keras
- Streamlit
- NumPy, Matplotlib, scikit-learn
- Dataset from Kaggle

---
###  Model Training

The `plant_disease.py` script handles the full training pipeline:

- **Builds multiple models**:
  - A custom Convolutional Neural Network (CNN)
  - Transfer learning models: MobileNetV2, VGG16, and ResNet50
- **Uses data generators** with real-time data augmentation to improve generalization:
  - Rotation, shift, shear, zoom, and horizontal flip
- **Trains each model** and evaluates using:
  - Accuracy, precision, recall, F1-score
- **Stores results**:
  - Trained models in `/models/` (`.h5` files)
  - Accuracy plots in `/plots/` (`.png` files)
  - Metrics reports in `/metrics_reports/` (JSON with per-class and macro-average metrics)
- **Uses callbacks**:
  - `EarlyStopping` to prevent overfitting
  - `ReduceLROnPlateau` to adjust learning rate if validation loss stops improving

 These artifacts help analyze and compare the performance of each model.

---

#### Clone this Repository

1. To get a local copy of this project, run the following command in your terminal:

```
git clone https://github.com/AshvinAK17/Final-Projects-Repo.git
cd Final-Projects-Repo/Plant_disease_detection

```

2. Create Virtual Environment (Optional but Recommended)

# For Windows
```
python -m venv env
env\Scripts\activate
```

# For macOS / Linux
```
python3 -m venv env
source env/bin/activate
```

3. Install Requirements:

```
pip install -r requirements.txt
```

4. Run the streamlit app:

```
streamlit run plant_disease_streamlit.py

```

### Streamlit:
You can also directly access the aapp through the below link:
**https://final-projects-repo-jsqkkdwmuhpu45wewzjvgy.streamlit.app/**

# Usage:
1. Upload a plant leaf image (.jpg, .jpeg, .png).
2. The app predicts:
    - Class name (e.g., Tomato___Leaf_Mold)
    - Confidence score (e.g., 92.4%)

Note: This application is developed for educational and project purposes.
Accuracy depends on dataset quality and model complexity.
