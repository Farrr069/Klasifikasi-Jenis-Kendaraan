# streamlit_app.py

import streamlit as st
import torch
import torchvision.transforms as transforms
from PIL import Image
from torchvision import models
from efficientnet_pytorch import EfficientNet
import torch.nn as nn
import os
import matplotlib.pyplot as plt
import seaborn as sns
import random
import math

# -----------------------------
# CONFIG
# -----------------------------
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
CLASS_NAMES = ['Auto Rickshaws', 'Bikes', 'Cars', 'Motorcycles', 'Planes', 'Ships', 'Trains']
IMG_SIZE = 128
EDA_DATA_DIR = os.path.join("dataset", "Vehicles")

# -----------------------------
# TRANSFORM
# -----------------------------
transform = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# -----------------------------
# CNN MODEL CLASS
# -----------------------------
class DeeperCNN(nn.Module):
    def __init__(self, num_classes):
        super(DeeperCNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(256 * 8 * 8, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

# -----------------------------
# LOAD MODELS
# -----------------------------
@st.cache_resource
def load_model(model_type):
    if model_type == "MobileNetV2":
        model = models.mobilenet_v2(pretrained=False)
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load("mobilenet_model.pth", map_location=DEVICE))

    elif model_type == "EfficientNet-B0":
        model = EfficientNet.from_name('efficientnet-b0')
        model._fc = nn.Linear(model._fc.in_features, len(CLASS_NAMES))
        model.load_state_dict(torch.load("efficientnet_model.pth", map_location=DEVICE))

    elif model_type == "Custom CNN":
        model = DeeperCNN(num_classes=len(CLASS_NAMES))
        model.load_state_dict(torch.load("cnn_model_complete.pth", map_location=DEVICE))

    else:
        st.error("Model tidak dikenali.")
        return None

    model.eval()
    return model.to(DEVICE)

# -----------------------------
# PREDICTION
# -----------------------------
def predict(image, model):
    img = transform(image).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        outputs = model(img)
        prob = torch.nn.functional.softmax(outputs, dim=1)
        conf, pred = torch.max(prob, 1)
        return CLASS_NAMES[pred.item()], conf.item() * 100

# -----------------------------
# EDA (EXPLORATORY DATA ANALYSIS)
# -----------------------------
def show_eda(data_dir=EDA_DATA_DIR):
    st.subheader("📊 Distribusi & Contoh Data per Kelas")

    if not os.path.exists(data_dir):
        st.warning(f"📁 Folder '{data_dir}' tidak ditemukan.")
        return

    # 1. Hitung jumlah gambar
    class_counts = {}
    for cls in os.listdir(data_dir):
        cls_path = os.path.join(data_dir, cls)
        if os.path.isdir(cls_path):
            count = len([f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))])
            class_counts[cls] = count

    if not class_counts:
        st.info("❗ Tidak ada gambar ditemukan untuk ditampilkan.")
        return

    # 2. Barplot jumlah gambar
    st.markdown("### 📦 Jumlah Gambar per Kelas")
    fig1, ax1 = plt.subplots()
    sns.barplot(x=list(class_counts.keys()), y=list(class_counts.values()), palette="pastel", ax=ax1)
    ax1.set_xlabel("Kelas")
    ax1.set_ylabel("Jumlah Gambar")
    ax1.set_title("Distribusi Gambar")
    plt.xticks(rotation=45)
    st.pyplot(fig1)

    # 3. Contoh gambar
    st.markdown("### 🖼️ Contoh Gambar Acak Tiap Kelas")
    cols = 3
    rows = math.ceil(len(class_counts) / cols)
    fig2, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    axes = axes.flatten()

    for idx, cls in enumerate(class_counts):
        class_path = os.path.join(data_dir, cls)
        images = [f for f in os.listdir(class_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))]
        if images:
            img_path = os.path.join(class_path, random.choice(images))
            img = Image.open(img_path)
            axes[idx].imshow(img)
            axes[idx].set_title(cls)
            axes[idx].axis('off')

    for j in range(idx + 1, len(axes)):
        axes[j].axis('off')

    plt.tight_layout()
    st.pyplot(fig2)

    # 4. Ukuran gambar
    st.markdown("### 📐 Ukuran Gambar Asli (1 contoh per kelas)")
    for cls in class_counts:
        cls_path = os.path.join(data_dir, cls)
        sample_img = [f for f in os.listdir(cls_path) if f.lower().endswith(('.jpg', '.png', '.jpeg'))][0]
        with Image.open(os.path.join(cls_path, sample_img)) as img:
            st.write(f"- **{cls}**: {img.size[0]} x {img.size[1]} (lebar x tinggi)")

# -----------------------------
# STREAMLIT UI
# -----------------------------
st.title("🚗🚲 Vehicle Image Classifier")

menu = st.sidebar.radio("📌 Navigasi", ["📤 Prediksi Gambar", "📈 EDA (Analisis Data)"])

if menu == "📈 EDA (Analisis Data)":
    show_eda()

elif menu == "📤 Prediksi Gambar":
    st.write("Upload gambar kendaraan dan pilih model untuk memprediksi jenisnya.")

    uploaded_file = st.file_uploader("📤 Upload Gambar", type=['jpg', 'png', 'jpeg'])
    model_type = st.selectbox("🧠 Pilih Model", ["MobileNetV2", "EfficientNet-B0", "Custom CNN"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file).convert("RGB")
        st.image(image, caption="🖼️ Gambar yang diunggah", width=700)

        if st.button("🔍 Prediksi"):
            model = load_model(model_type)
            pred_class, confidence = predict(image, model)
            st.success(f"✅ Prediksi: **{pred_class}**")
            st.info(f"📊 Keyakinan: {confidence:.2f}%")
