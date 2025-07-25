import os
import json
import torch
import numpy as np
from PIL import Image
import streamlit as st
from facenet_pytorch import MTCNN
import mediapipe as mp
from torchvision import transforms
import torch.nn as nn

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load model
class EmotionCNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3,64,3,padding=1), nn.BatchNorm2d(64), nn.LeakyReLU(0.1), nn.MaxPool2d(2), nn.Dropout(0.3),
            nn.Conv2d(64,128,3,padding=1), nn.BatchNorm2d(128), nn.LeakyReLU(0.1), nn.MaxPool2d(2), nn.Dropout(0.4),
            nn.Conv2d(128,256,3,padding=1), nn.BatchNorm2d(256), nn.LeakyReLU(0.1), nn.AdaptiveAvgPool2d((4,4)), nn.Dropout(0.4)
        )
        self.landmark_mlp = nn.Sequential(
            nn.Linear(1404, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(256*4*4 + 256, 256), nn.BatchNorm1d(256), nn.LeakyReLU(0.1), nn.Dropout(0.4),
            nn.Linear(256,128), nn.BatchNorm1d(128), nn.LeakyReLU(0.1), nn.Dropout(0.3),
            nn.Linear(128,7)
        )
    def forward(self, x, landmarks):
        x = self.features(x)
        x = torch.flatten(x,1)
        l = self.landmark_mlp(landmarks)
        x = torch.cat([x, l], dim=1)
        return self.classifier(x)

model = EmotionCNN().to(device)
model.load_state_dict(torch.load("Emotion Detection/ed_final_one_best_model.pth", map_location=device))
model.eval()

# Load artifacts
mtcnn = MTCNN(image_size=128, margin=10, keep_all=False, device=device)
mp_face = mp.solutions.face_mesh
lm_mean = torch.load("Emotion Detection/lm_mean.pt")
lm_std  = torch.load("Emotion Detection/lm_std.pt")
with open("Emotion Detection/classes.json", "r") as f:
    classes = json.load(f)

base_transform = transforms.Compose([
    transforms.Resize((128,128)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5,0.5,0.5], std=[0.5,0.5,0.5])
])

# ----------------------------
# Prediction function
# ----------------------------
def predict_emotion(image):
    face_tensor = mtcnn(image)
    if isinstance(face_tensor, list):
        face_tensor = face_tensor[0] if len(face_tensor) > 0 else None
    if face_tensor is None:
        face_tensor = base_transform(image).to(device).unsqueeze(0)
    else:
        face_tensor = base_transform(transforms.ToPILImage()(face_tensor)).to(device).unsqueeze(0)

    # Landmarks
    with mp_face.FaceMesh(static_image_mode=True, max_num_faces=1) as fm:
        np_face = (face_tensor[0].permute(1,2,0).cpu().numpy() * 255).astype(np.uint8)
        results = fm.process(np_face)
        if results.multi_face_landmarks:
            landmarks = [(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark]
            if len(landmarks)==468:
                landmarks = torch.tensor(np.array(landmarks).flatten(), dtype=torch.float)
                landmarks = (landmarks - lm_mean) / lm_std
                landmarks = landmarks.unsqueeze(0).to(device)
            else:
                landmarks = lm_mean.unsqueeze(0).to(device)
        else:
            landmarks = lm_mean.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(face_tensor, landmarks)
        probs = torch.softmax(outputs, dim=1).cpu().numpy()[0]
        top_idx = np.argmax(probs)
        predicted_class = classes[top_idx]
    return predicted_class, probs

# ----------------------------
# Streamlit UI
# ----------------------------
st.set_page_config(page_title="Emotion Detection App", layout="centered")
st.title(" Emotion Detection App")
st.write("Upload an image and see the predicted emotion and scores.")

uploaded_file = st.file_uploader("Choose an image...", type=["jpg","jpeg","png"])

if uploaded_file:
    try:
        uploaded_file.seek(0, os.SEEK_END)
        file_size = uploaded_file.tell()
        uploaded_file.seek(0)
        if file_size > 5*1024*1024:
            st.error("File too large (>5MB).")
        else:
            image = Image.open(uploaded_file).convert("RGB")
            st.image(image, caption="Uploaded Image", use_container_width =True)

            # Immediately predict
            predicted_class, probs = predict_emotion(image)

            st.success(f"Predicted Emotion: **{predicted_class}**")

            # Show scores
            st.subheader("Prediction Scores:")
            prob_table = {cls: f"{100*p:.2f}%" for cls, p in sorted(zip(classes, probs), key=lambda x: -x[1])}
            st.table(prob_table)

    except Exception as e:
        st.error(f"Error: {e}")
