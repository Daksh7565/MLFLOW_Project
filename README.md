# 🌾 **MLOps Project: End-to-End Crop Disease Classification**

1 A production-ready MLOps system for automating the training, evaluation, deployment, and monitoring of deep learning models for **crop disease classification**.
The system integrates **ZenML**, **MLflow**, and **Streamlit** to deliver a complete and scalable pipeline from data ingestion to real-time inference.

---

## 🧭 **Table of Contents**

* [✨ Features](#-features)
* [🛠️ Tech Stack](#️-tech-stack)
* [📂 Project Structure](#-project-structure)
* [⚙️ Setup & Installation](#️-setup--installation)
* [🚀 How to Run the Project](#-how-to-run-the-project)

  * Start MLflow Tracking Server
  * Run a Training Pipeline
  * Launch the Streamlit App
* [🏗️ Pipeline Architecture](#️-pipeline-architecture)
* [🧠 Model Details](#-model-details)
* [🌐 Using the Web App](#-using-the-web-app)

---

## ✨ **Features**

✔ **Automated and reproducible MLOps pipelines** using ZenML
✔ **Two model architectures** — a standard CNN and an advanced Hybrid CNN
✔ **MLflow-powered experiment tracking** for metrics, artifacts, and model history
✔ **Automatic model registration** in MLflow Model Registry
✔ **Clean Streamlit UI** for real-time disease prediction
✔ **Automatic dataset splitting** (train/validation/test)
✔ **Scalable, modular, production-ready design**

---

## 🛠️ **Tech Stack**

| Component                   | Technology         |
| --------------------------- | ------------------ |
| **MLOps Pipeline**          | ZenML              |
| **Experiment Tracking**     | MLflow             |
| **Deep Learning Framework** | TensorFlow / Keras |
| **Web UI**                  | Streamlit          |
| **Image Processing**        | Pillow, OpenCV     |
| **Data Handling**           | Pandas, NumPy      |
| **Data Splitting**          | split-folders      |

---

## 📂 **Project Structure**

```
.
├── Main dataset/                 # Raw images (each class in a subfolder)
├── output/                       # Train/Val/Test split data
├── pipelines/
│   ├── cnn.py                    # Standard CNN pipeline
│   └── hybrid_cnn.py             # Hybrid model pipeline
├── steps/
│   ├── data.py                   # Dataset splitter
│   ├── model.py                  # Standard CNN trainer
│   ├── Model2.py                 # Hybrid model trainer
│   └── evalute.py                # Evaluation + MLflow logging
├── main.py                       # Executes the hybrid pipeline
├── streamlit.py                  # Streamlit prediction app
└── Requirements.txt              # Dependencies
```

---

## ⚙️ **Setup & Installation**

### **1. Prerequisites**

* Python **3.8+**
* Virtual environment (**venv** / **conda** recommended)

---

### **2. Clone the Repository**

```bash
git clone <your-repository-url>
cd <your-repository-name>
```

---

### **3. Create a Virtual Environment**

```bash
python -m venv venv
source venv/bin/activate       # Windows: venv\Scripts\activate
```

---

### **4. Install Dependencies**

```bash
pip install -r Requirements.txt
```

---

### **5. Prepare Your Dataset**

* Create a folder named **Main dataset**
* Inside it, create subfolders like:

  * `Healthy`
  * `Bacterial Blight`
  * `Aphids`
  * ...etc.
* Place images inside corresponding class folders.

---

## 🚀 **How to Run the Project**

### 🔹 **1. Start MLflow Tracking Server**

In a separate terminal:

```bash
mlflow ui --port 8080
```

Open: **[http://127.0.0.1:8080](http://127.0.0.1:8080)**

---

### 🔹 **2. Run the Training Pipeline**

The `main.py` is configured for the hybrid model pipeline.

```bash
python main.py
```

This will:

1. Set MLflow experiment name → **"Stark"**
2. Split the dataset into `output/`
3. Train the base CNN
4. Train the Hybrid CNN (CNN + MobileNet + ResNet50)
5. Evaluate and log everything to MLflow
6. Register the final model in the MLflow Model Registry

---

### 🔹 **3. Launch the Streamlit App**

After training completes:

```bash
streamlit run streamlit.py
```

A browser tab will open with the **Crop Disease Predictor UI**.

---

## 🏗️ **Pipeline Architecture**

ZenML automates all workflow stages via modular steps:

### **🟦 1. `data_splitter` (data.py)**

* Reads raw image dataset
* Splits into **80% Train / 10% Val / 10% Test**
* Saves to the output directory

---

### **🟩 2. `cnn_trainer` (model.py)**

* Builds a foundational CNN
* Logs:

  * Parameters
  * Training curves
  * Metrics
* Returns trained CNN + history

---

### **🟨 3. `Hybrid_model` (Model2.py)**

Used only in the hybrid pipeline.

Model includes three feature extraction branches:

* Base CNN (trained previously)
* Pretrained **MobileNet**
* Pretrained **ResNet50**

All features are concatenated → passed into dense classifier → trained with logs.

---

### **🟥 4. `evaluator` (evalute.py)**

* Runs evaluation on test data
* Logs:

  * Final accuracy & loss
  * CSV training history
  * Learning curves
* Registers the final model in MLflow Model Registry
  → Name: **`hybrid_image_classifier`**

---

## 🧠 **Model Details**

### **🔹 Base CNN**

A simple sequential architecture with:

* Conv + MaxPool layers
* Flatten + Dense classifier
* Serves as:

  * Baseline
  * Component for hybrid branch

---

### **🔹 Hybrid Model**

A high-performance ensemble combining:

| Branch        | Description                                              |
| ------------- | -------------------------------------------------------- |
| **Base CNN**  | Learns dataset-specific low-level features               |
| **MobileNet** | Lightweight pretrained transfer learning branch          |
| **ResNet50**  | Deep, powerful pretrained branch for high-level features |

All outputs are concatenated → Dense layers → Final classification.
This hybrid design increases accuracy by blending:

* Custom learning
* Lightweight transfer learning
* Deep pretrained representations

---

## 🌐 **Using the Web App**

1. Ensure Streamlit app is running
2. Open the UI
3. Upload a JPG/PNG leaf image
4. Click **"Classify Image"**

The app will display:

* ✅ Predicted class
* 📊 Confidence score
* 📉 Confidence distribution chart

Perfect for real-time deployment or field usage.

