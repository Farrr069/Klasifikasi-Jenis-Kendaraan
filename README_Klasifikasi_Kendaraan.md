
# 🚗 Vehicle Image Classification with CNN & Transfer Learning

## 📘 Deskripsi Project
Proyek ini merupakan sistem klasifikasi gambar kendaraan yang dibangun dengan tiga pendekatan model pembelajaran mesin berbasis citra:

1. **Model Neural Network Non-Pretrained (Custom CNN)**  
2. **Transfer Learning dengan MobileNetV2**
3. **Transfer Learning dengan EfficientNet-B0**

Aplikasi ini dirancang menggunakan **Streamlit** yang memungkinkan pengguna mengunggah gambar kendaraan dan memprediksi jenis kendaraan berdasarkan model yang dipilih.

---

## 📌 Latar Belakang
Klasifikasi objek dalam citra merupakan tantangan umum dalam computer vision. Dalam konteks kendaraan, pengenalan jenis kendaraan dari gambar bisa digunakan untuk sistem lalu lintas cerdas, inventaris otomatis, atau pengawasan.

Untuk meningkatkan akurasi klasifikasi, proyek ini menggabungkan model yang dilatih dari awal (**Custom CNN**) dengan dua model populer pretrained (**MobileNetV2** dan **EfficientNet-B0**) menggunakan pendekatan **Transfer Learning**.

---

## 🎯 Tujuan Pengembangan
- Membangun sistem klasifikasi kendaraan berbasis citra.
- Menggunakan pendekatan Custom CNN dan Transfer Learning.
- Menyediakan antarmuka interaktif berbasis web menggunakan Streamlit.
- Menyediakan analisis data (EDA) visual untuk memahami distribusi dataset.

---

## 📂 Sumber Dataset
Dataset disusun dalam struktur direktori seperti berikut:

```
dataset/
└── Vehicles/
    ├── Auto Rickshaws/
    ├── Bikes/
    ├── Cars/
    ├── Motorcycles/
    ├── Planes/
    ├── Ships/
    └── Trains/
```

Setiap subfolder berisi gambar dari masing-masing jenis kendaraan.

---

## 🧪 Preprocessing dan Pemodelan

### 🧾 Pemilihan Kolom/Atribut
- **Input**: Gambar kendaraan (.jpg, .png, .jpeg)
- **Target**: Kelas kendaraan (7 kelas)

### 🧹 Preprocessing Data
- Resize gambar ke `128x128`
- Normalisasi menggunakan mean dan std dari ImageNet
- Augmentasi (jika digunakan) dilakukan selama pelatihan

### 🤖 Pemodelan
1. **Custom CNN (DeeperCNN)**
   - Dibangun dari 4 blok konvolusi dan fully connected layer
   - Tanpa bobot pretrained
2. **MobileNetV2**
   - Transfer Learning dari PyTorch
   - Layer terakhir disesuaikan menjadi 7 output
3. **EfficientNet-B0**
   - Menggunakan pustaka `efficientnet_pytorch`
   - Fully connected layer diganti untuk 7 kelas output

---

## 🛠️ Langkah Instalasi

### 💻 Software Utama
- Python >= 3.8
- Streamlit
- PyTorch

### 📦 Dependensi
Instal semua dependensi dengan:

```bash
pip install -r requirements.txt
```

---

## ▶️ Menjalankan Sistem Prediksi

### 📁 Siapkan Dataset
Pastikan dataset diletakkan dalam folder `dataset/Vehicles/` dengan struktur seperti sebelumnya.

### 🚀 Jalankan Aplikasi

```bash
streamlit run streamlit_app.py
```

---

## 🏋️ Pelatihan Model
Model Custom CNN dilatih menggunakan arsitektur CNN dalam file `Notebook_Model.ipynb`. Setelah pelatihan, bobot disimpan menggunakan:

```python
torch.save(model.state_dict(), "cnn_model_complete.pth")
```

Model MobileNetV2 dan EfficientNet juga dilatih dengan Transfer Learning dan bobot masing-masing disimpan sebagai:

- `mobilenet_model.pth`
- `efficientnet_model.pth`

---

## 📊 Hasil dan Analisis

### 📈 Evaluasi Model
Evaluasi dilakukan menggunakan akurasi validasi dan metrik seperti:

- Confusion matrix
- Loss dan akurasi per epoch
- Prediksi top-1

Model Custom CNN menunjukkan performa yang kompetitif, namun model transfer learning umumnya memiliki akurasi lebih tinggi dan konvergensi lebih cepat.

---

## 💻 Sistem Sederhana Streamlit

Aplikasi Streamlit menyediakan dua fitur utama:

### ✅ Prediksi Gambar
- Upload gambar kendaraan
- Pilih model yang diinginkan
- Sistem menampilkan hasil prediksi dan confidence level

### 📊 EDA (Exploratory Data Analysis)
- Menampilkan jumlah gambar per kelas dalam bar chart
- Menampilkan gambar contoh per kelas
- Cek ukuran asli gambar

---

## 🖼️ Tampilan

![alt text](image.png)

---

## 🔗 Link Live Demo
💡 *Aplikasi ini belum dihosting, namun dapat dijalankan secara lokal menggunakan perintah `streamlit run`.*

---

## 👤 Biodata

**Nama**: Farrel Maulana Irfan  
**Jurusan**: Informatika  
**Fokus**: Machine Learning & Deep Learning  
**Tahun**: 2025  
**Proyek**: UAP Klasifikasi Citra Kendaraan  
