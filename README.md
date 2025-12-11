# Drone Telemetry Analysis Pipeline

## Overview
This repository provides a complete analytical workflow for **drone telemetry malfunction detection**, including:

- Dataset extraction and auto-discovery
- Full Exploratory Data Analysis (EDA)
- Preprocessing (missing data handling, outlier treatment, feature engineering, scaling)
- Model development:
  - **LSTM** for sequence modeling
  - **SVM** for classical ML classification
  - **VAE** (Variational Autoencoder) for anomaly detection
- Hyperparameter tuning (GridSearch, RandomizedSearch, Keras-Tuner)
- Model evaluation and comparison
- Explainability (XAI): SHAP, LIME, PDP
- Model saving & reproducibility utilities

This README provides installation instructions, environment setup, project structure, detailed pipeline description, and usage guidelines following industry-standard README formats.

---

# Table of Contents
1. [Project Description](#project-description)
2. [Key Features](#key-features)
3. [Dataset](#dataset)
4. [Project Structure](#project-structure)
5. [Environment Setup](#environment-setup)
6. [Installation](#installation)
7. [Usage Guide](#usage-guide)
8. [Modeling Pipeline](#modeling-pipeline)
9. [Explainability (XAI)](#explainability-xai)
10. [Results](#results)
11. [Troubleshooting](#troubleshooting)
12. [Future Improvements](#future-improvements)
13. [License](#license)
14. [Contact](#contact)

---

# Project Description
This project focuses on analyzing drone telemetry data to detect **malfunctions, DoS attacks, and normal operational states**. The dataset contains 79+ sensor channels spanning: GPS, IMU, battery, CPU, RC-out, velocity, position, and custom drone state messages.

The goal is to:
- Clean and preprocess raw telemetry data
- Understand drone behavior under malfunction conditions
- Train ML/DL models to classify and detect anomalies
- Use explainability tools to analyze model decisions

---

# Key Features
- **Automatic ZIP extraction & CSV discovery**
- **High-quality visual EDA**
- **Advanced missing data strategy** (interpolation, median imputation)
- **Feature engineering** (velocity magnitude, distance to target, battery drain, timestamp features)
- **Two scaling pipelines** for SVM & LSTM
- **Three modeling techniques**: LSTM, SVM, VAE
- **Hyperparameter tuning** using industry-standard techniques
- **Explainability toolkit**: SHAP, LIME, PDP
- **Export-ready model saving**

---

# Dataset
Place your dataset ZIP file in the root directory and update the path in the notebook:

```
zip_file_path = "/content/Hand on ML - Assignemnt.zip"
```

The extraction directory is automatically created:
```
/content/extracted_zip
```

The notebook automatically finds the correct telemetry CSV inside nested folders.

---

# Project Structure
```
project-root/
│
├── notebooks/
│   └── drone_telemetry_analysis.ipynb
```

---

# Environment Setup
Ensure you are using:

- Python 3.9+
- TensorFlow 2.12+
- scikit-learn, numpy, pandas, matplotlib, seaborn
- SHAP & LIME for explainability

### Recommended Environment
Use **Google Colab** or a local machine with a **GPU**.

---

# Installation
### 1. Clone the Repository
```
git clone https://github.com/<your-username>/<your-repo>.git
cd <your-repo>
```

### 2. Create a Virtual Environment
```
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

# Usage Guide
### 1. Upload your dataset ZIP into the notebook environment
### 2. Run the extraction cell:
```
with zipfile.ZipFile(zip_file_path, 'r') as zip_ref:
    zip_ref.extractall(extraction_dir)
```

### 3. Run the entire notebook step-by-step
The notebook includes:
- Data loading
- EDA
- Preprocessing
- Feature engineering
- Model training
- Evaluation
- Explainability (XAI)
- Model saving
- 
---

# Modeling Pipeline
## 1. Preprocessing
- Missing value handling using median and interpolation
- Outlier capping using IQR
- Removal of low-information or high-missing columns
- Feature scaling using StandardScaler and MinMaxScaler

## 2. Feature Engineering
- Velocity magnitude
- Distance-to-target via haversine formula
- Battery drain rate
- Timestamp-derived features (hour, elapsed seconds)

## 3. Models
### (A) LSTM Deep Learning Model
- Sequence windowing
- Stacked LSTM layers
- Dropout regularization
- Adam optimizer with tuned learning rate

### (B) Support Vector Machine (SVM)
- RBF kernel
- Grid & Randomized search
- StandardScaler normalization

### (C) Variational Autoencoder (VAE)
- Reconstruction error thresholding
- Anomaly detection
- Best for rare malfunction events

## 4. Hyperparameter Tuning
- **GridSearchCV** for SVM
- **RandomizedSearchCV** for fast search
- **Keras-Tuner** for LSTM

---

# Explainability (XAI)
The notebook includes:

## SHAP
- Global feature importance
- Force plots
- Decision plots

## LIME
- Local instance explanations
- Feature contribution overview

## PDP (Partial Dependence Plots)
- Feature impact on model output

---

# Results
Results vary per dataset, but typical outputs include:
- Precision, Recall, F1-score per class
- Confusion matrix
- ROC curves
- SHAP summary plots
- LIME local explanations

All trained models are saved in the `models/` directory.

---

# Troubleshooting
### 1. CSV Not Found
Ensure your ZIP file contains a `.csv` file. The notebook auto-detects it.

### 2. TensorFlow Version Conflict
Update TF:
```
pip install tensorflow==2.12.0
```

### 3. GPU Not Detected
Enable GPU via Colab > Runtime > Change Runtime > GPU.

### 4. Memory Issues
Reduce batch size or window size for LSTM.

---

# Future Improvements
- Deploy the model as an API for real-time drone monitoring
- Add more anomaly detection models (Isolation Forest, Deep SVDD)
- Create dashboards using Streamlit
- Add multi-drone telemetry ingestion

---

# License
This project is licensed under the MIT License. See `LICENSE` for details.

---

# Contact
**Developer:** Muhammad Talha  
**Institute:** FAST - NUCES  
For queries, collaborations, or enterprise solutions:  
`muhammedtalha81@example.com`
