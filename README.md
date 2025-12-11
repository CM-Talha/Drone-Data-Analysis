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
11. [Loading and Using Trained Models](#loading-and-using-trained-models)
12. [Troubleshooting](#troubleshooting)
13. [Future Improvements](#future-improvements)
14. [License](#license)
15. [Contact](#contact)

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
Drone-Data-Analysis/
│
├── README.md                          # Project documentation
├── LICENSE                            # MIT License
├── notebook/
│   └── Drone_Data_Analysis_K25_7605.ipynb    # Main analysis notebook
├── pre-processed-data/
│   ├── combined_df_clean.csv          # Cleaned and processed telemetry data
│   └── preprocessed_drone_data.pkl    # Serialized preprocessed data
├── trained_models/
│   ├── lstm_final.joblib              # Final LSTM model
│   ├── final_svm_model.pkl            # Final SVM classifier
│   ├── svm_model.pkl                  # SVM model (alternative version)
│   ├── vae_model.keras                # Complete VAE model
│   ├── vae_encoder.keras              # VAE encoder component
│   └── vae_decoder.keras              # VAE decoder component
├── reports/
│   ├── Comprehensive Analytical Report on Drone Telemetry Data.pdf
│   └── images/                        # Generated plots and visualizations
└── code_and_trained_models.zip        # Archived codebase and models
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
```bash
git clone https://github.com/CM-Talha/Drone-Data-Analysis.git
cd Drone-Data-Analysis
```

### 2. Create a Virtual Environment
```bash
python3 -m venv venv
source venv/bin/activate       # Linux/Mac
venv\Scripts\activate          # Windows
```

### 3. Install Dependencies
```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Alternatively, install specific packages:
```bash
pip install tensorflow==2.19.0 tf-keras==2.19.0 scikit-learn pandas numpy
```

# Usage Guide

## Running the Analysis

### 1. Navigate to the Notebook
Open the main analysis notebook:
```
notebook/Drone_Data_Analysis_K25_7605.ipynb
```

### 2. Install Dependencies
The notebook includes a dependency installation cell that installs all required packages:
```
tensorflow==2.19.0
tf-keras==2.19.0
tensorflow-text==2.19.0
tensorflow-decision-forests==1.12.0
pandas==2.2.2
numpy, opencv-python, numba, and others
```

### 3. Run the Notebook
Execute cells sequentially to:
- Load and explore the preprocessed telemetry data
- Generate exploratory data analysis (EDA) visualizations
- Preprocess and engineer features
- Train and evaluate ML/DL models
- Apply explainability techniques (SHAP, LIME)
- Save trained models

### 4. Access Trained Models
All pre-trained models are available in `trained_models/`:
- Load and use for predictions without retraining
- Evaluate on new telemetry data
---

# Modeling Pipeline

## Complete Analysis Workflow
The notebook implements a comprehensive ML/DL pipeline for drone telemetry malfunction detection:

### 1. Data Exploration & Analysis
- Summary statistics and distributions
- Missing data analysis
- Sensor correlation analysis
- Time-series visualization
- Multivariate analysis

### 2. Preprocessing
- **Missing Value Handling**: Median imputation and interpolation
- **Outlier Treatment**: IQR-based capping
- **Column Filtering**: Removal of low-information or high-missing columns
- **Feature Scaling**: StandardScaler and MinMaxScaler for different models

### 3. Feature Engineering
- Velocity magnitude calculation
- Distance-to-target via haversine formula
- Battery drain rate estimation
- Temporal features (hour of day, elapsed seconds)
- Derived telemetry indicators

### 4. Model Development & Training

#### (A) LSTM (Long Short-Term Memory) - Deep Learning
- **Location:** `trained_models/lstm_final.joblib`
- Sequence windowing for temporal patterns
- Stacked LSTM layers with dropout regularization
- Adam optimizer with learning rate tuning
- Best for: Sequential pattern detection and temporal anomalies

#### (B) Support Vector Machine (SVM) - Classical ML
- **Locations:** `trained_models/final_svm_model.pkl`, `trained_models/svm_model.pkl`
- RBF (Radial Basis Function) kernel
- Hyperparameter tuning: GridSearchCV and RandomizedSearchCV
- StandardScaler normalization
- Best for: High-dimensional classification with clear decision boundaries

#### (C) Variational Autoencoder (VAE) - Deep Generative Model
- **Locations:** `trained_models/vae_model.keras`, `trained_models/vae_encoder.keras`, `trained_models/vae_decoder.keras`
- Reconstruction error-based anomaly detection
- Encoder learns compressed representations
- Decoder reconstructs telemetry sequences
- Best for: Detecting rare malfunction events and novel anomalies

### 5. Hyperparameter Tuning
- **GridSearchCV**: Exhaustive parameter search for SVM
- **RandomizedSearchCV**: Fast stochastic search for large parameter spaces
- **Keras-Tuner**: Automated hyperparameter optimization for LSTM

### 6. Model Evaluation
- Classification metrics: Precision, Recall, F1-score per class
- Confusion matrices for all models
- ROC curves and AUC analysis
- Cross-validation scoring
- Comparison of model performance

---

# Explainability & Interpretability (XAI)

The notebook implements multiple explainability techniques to understand model decisions:

## SHAP (SHapley Additive exPlanations)
- **Global Feature Importance**: Identifies features most influential across all predictions
- **Force Plots**: Shows how each feature contributes to individual predictions
- **Decision Plots**: Visualizes model logic flow for specific instances
- **Summary Plots**: Aggregated feature impact analysis

## LIME (Local Interpretable Model-agnostic Explanations)
- **Local Instance Explanations**: Explains individual predictions in local feature space
- **Feature Contribution Overview**: Shows which features are most important for specific decisions
- Model-agnostic approach works with LSTM, SVM, and VAE

## PDP (Partial Dependence Plots)
- **Feature Impact Analysis**: Shows how individual features affect model predictions
- **Marginal Effects**: Visualizes relationships between features and outputs
- Helps identify feature interactions and non-linear patterns

## Model-Specific Insights
- **LSTM**: Temporal attention patterns and sequence dependencies
- **SVM**: Support vector distribution and decision boundaries
- **VAE**: Reconstruction quality and latent space structure

---

# Results & Outputs

## Generated Artifacts
All analysis results are saved in the project:

### Trained Models (Production-Ready)
Located in `trained_models/`:
- `lstm_final.joblib` - LSTM model for sequence classification
- `final_svm_model.pkl` - Final tuned SVM classifier
- `svm_model.pkl` - Alternative SVM implementation
- `vae_model.keras` - Complete VAE for anomaly detection
- `vae_encoder.keras` - Encoder for dimensionality reduction
- `vae_decoder.keras` - Decoder for reconstruction

### Reports
Located in `reports/`:
- **Comprehensive Analytical Report on Drone Telemetry Data.pdf** - Full analysis summary with findings and recommendations
- **images/** - Directory containing generated visualizations:
  - EDA plots (distributions, correlations, time-series)
  - Model performance visualizations
  - Feature importance plots (SHAP, LIME)
  - Confusion matrices
  - ROC curves and evaluation metrics

### Model Performance Metrics
- **Accuracy, Precision, Recall, F1-score** per class for each model
- **Confusion Matrix** showing classification breakdown
- **ROC Curves** with AUC scores for binary and multi-class scenarios
- **Cross-validation** results with confidence intervals
- **SHAP Summary Plots** highlighting top feature contributors
- **LIME Local Explanations** for individual prediction understanding

### Key Findings
- Identification of critical sensor channels for malfunction detection
- Performance comparison across LSTM, SVM, and VAE models
- Anomaly detection thresholds for real-time applications
- Feature engineering impact on model improvements

---

# Loading and Using Trained Models

This section provides step-by-step instructions and code snippets for loading each trained model and making predictions on preprocessed drone telemetry data.

## Prerequisites

```python
import pandas as pd
import numpy as np
import joblib
import pickle
import tensorflow as tf
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import warnings
warnings.filterwarnings('ignore')

# Set paths
MODEL_DIR = 'trained_models/'
DATA_DIR = 'pre-processed-data/'
```

## Loading Preprocessed Data

### Option 1: Load from CSV
```python
# Load cleaned telemetry data
data = pd.read_csv(f'{DATA_DIR}combined_df_clean.csv')
print(f"Data shape: {data.shape}")
print(f"Columns: {data.columns.tolist()}")
```

### Option 2: Load from Pickle
```python
# Load serialized preprocessed data
with open(f'{DATA_DIR}preprocessed_drone_data.pkl', 'rb') as f:
    data = pickle.load(f)
print(f"Data shape: {data.shape}")
```

---

## Model 1: LSTM (Sequence Classification)

### Load the Model
```python
# Load LSTM model using joblib
lstm_model = joblib.load(f'{MODEL_DIR}lstm_final.joblib')
print("LSTM model loaded successfully!")
print(f"Model type: {type(lstm_model)}")
```

### Prepare Data for LSTM

LSTM models expect 3D input: `(batch_size, sequence_length, features)`

```python
def prepare_lstm_data(data, sequence_length=50, scaler=None):
    """
    Prepare data for LSTM prediction.
    
    Args:
        data: DataFrame with features
        sequence_length: Number of time steps for each sequence (default: 50)
        scaler: StandardScaler instance (if None, creates new one)
    
    Returns:
        sequences: 3D array (n_sequences, sequence_length, n_features)
        scaler: Fitted scaler for inverse transformation
    """
    # Extract features (exclude target if present)
    if 'label' in data.columns or 'target' in data.columns:
        features = data.drop(['label', 'target'], axis=1, errors='ignore')
    else:
        features = data
    
    # Scale data
    if scaler is None:
        scaler = StandardScaler()
        scaled_data = scaler.fit_transform(features)
    else:
        scaled_data = scaler.transform(features)
    
    # Create sequences
    sequences = []
    for i in range(len(scaled_data) - sequence_length + 1):
        sequences.append(scaled_data[i:i + sequence_length])
    
    return np.array(sequences), scaler

# Prepare data
lstm_sequences, lstm_scaler = prepare_lstm_data(data, sequence_length=50)
print(f"Sequences shape: {lstm_sequences.shape}")
```

### Make Predictions with LSTM
```python
# Get predictions
lstm_predictions = lstm_model.predict(lstm_sequences)

# For classification (if output is one-hot encoded or probabilities)
if lstm_predictions.ndim > 1 and lstm_predictions.shape[1] > 1:
    predicted_classes = np.argmax(lstm_predictions, axis=1)
    confidence_scores = np.max(lstm_predictions, axis=1)
else:
    predicted_classes = (lstm_predictions > 0.5).astype(int).flatten()
    confidence_scores = lstm_predictions.flatten()

print(f"Predictions shape: {lstm_predictions.shape}")
print(f"Sample predictions (first 5): {predicted_classes[:5]}")
print(f"Sample confidence scores: {confidence_scores[:5]}")
```

---

## Model 2: Support Vector Machine (SVM)

### Load the Model

```python
# Load SVM model (using final_svm_model.pkl)
with open(f'{MODEL_DIR}final_svm_model.pkl', 'rb') as f:
    svm_model = pickle.load(f)
print("SVM model loaded successfully!")

# Alternative: Load svm_model.pkl
# with open(f'{MODEL_DIR}svm_model.pkl', 'rb') as f:
#     svm_model = pickle.load(f)
```

### Prepare Data for SVM

```python
def prepare_svm_data(data, scaler=None):
    """
    Prepare data for SVM prediction.
    SVM requires 2D input: (n_samples, n_features)
    
    Args:
        data: DataFrame with features
        scaler: StandardScaler instance (if None, creates new one)
    
    Returns:
        features_scaled: 2D array (n_samples, n_features)
        scaler: Fitted scaler
    """
    # Extract features
    if 'label' in data.columns or 'target' in data.columns:
        features = data.drop(['label', 'target'], axis=1, errors='ignore')
    else:
        features = data
    
    # Scale data (SVM is sensitive to feature scaling)
    if scaler is None:
        scaler = StandardScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)
    
    return features_scaled, scaler

# Prepare data
svm_features, svm_scaler = prepare_svm_data(data)
print(f"Features shape: {svm_features.shape}")
```

### Make Predictions with SVM

```python
# Get predictions and decision scores
svm_predictions = svm_model.predict(svm_features)
svm_decision_scores = svm_model.decision_function(svm_features)

print(f"Predictions shape: {svm_predictions.shape}")
print(f"Sample predictions (first 5): {svm_predictions[:5]}")
print(f"Sample decision scores (first 5): {svm_decision_scores[:5]}")

# Get probabilities (if probability=True in training)
if hasattr(svm_model, 'predict_proba'):
    svm_probabilities = svm_model.predict_proba(svm_features)
    print(f"Probabilities shape: {svm_probabilities.shape}")
else:
    print("Model does not support probability estimates")
```

---

## Model 3: Variational Autoencoder (VAE)

### Load VAE Components

```python
# Load complete VAE model
vae_model = tf.keras.models.load_model(f'{MODEL_DIR}vae_model.keras')
print("VAE model loaded successfully!")

# Load encoder (for dimensionality reduction)
vae_encoder = tf.keras.models.load_model(f'{MODEL_DIR}vae_encoder.keras')
print("VAE encoder loaded successfully!")

# Load decoder (for reconstruction)
vae_decoder = tf.keras.models.load_model(f'{MODEL_DIR}vae_decoder.keras')
print("VAE decoder loaded successfully!")

# Optional: Print model summaries
print("\nVAE Model Summary:")
vae_model.summary()
```

### Prepare Data for VAE

```python
def prepare_vae_data(data, scaler=None):
    """
    Prepare data for VAE prediction.
    VAE requires 2D input: (n_samples, n_features)
    
    Args:
        data: DataFrame with features
        scaler: MinMaxScaler instance (VAE typically uses normalized data [0,1])
    
    Returns:
        features_scaled: 2D array (n_samples, n_features)
        scaler: Fitted scaler
    """
    # Extract features
    if 'label' in data.columns or 'target' in data.columns:
        features = data.drop(['label', 'target'], axis=1, errors='ignore')
    else:
        features = data
    
    # Scale data to [0, 1] range (typical for VAE)
    if scaler is None:
        scaler = MinMaxScaler()
        features_scaled = scaler.fit_transform(features)
    else:
        features_scaled = scaler.transform(features)
    
    return features_scaled, scaler

# Prepare data
vae_features, vae_scaler = prepare_vae_data(data)
print(f"Features shape: {vae_features.shape}")
```

### Make Predictions and Detect Anomalies with VAE

```python
# Get reconstruction from complete VAE model
vae_reconstructions = vae_model.predict(vae_features)

# Calculate reconstruction error (for anomaly detection)
reconstruction_error = np.mean(np.square(vae_features - vae_reconstructions), axis=1)

# Define anomaly threshold (can be adjusted based on percentile)
threshold = np.percentile(reconstruction_error, 95)  # Top 5% as anomalies
anomaly_predictions = (reconstruction_error > threshold).astype(int)

print(f"Reconstructions shape: {vae_reconstructions.shape}")
print(f"Reconstruction error - Mean: {reconstruction_error.mean():.4f}, Std: {reconstruction_error.std():.4f}")
print(f"Anomaly threshold: {threshold:.4f}")
print(f"Anomalies detected: {anomaly_predictions.sum()} out of {len(anomaly_predictions)}")

# Use encoder for dimensionality reduction
encoded_features = vae_encoder.predict(vae_features)
print(f"\nEncoded features shape: {encoded_features.shape}")

# Use decoder to reconstruct from encoded representation
decoded_features = vae_decoder.predict(encoded_features)
print(f"Decoded features shape: {decoded_features.shape}")
```

### Anomaly Detection Workflow

```python
# Complete anomaly detection pipeline using VAE
def detect_anomalies_vae(data, model, scaler, threshold_percentile=95):
    """
    Detect anomalies using VAE reconstruction error.
    
    Args:
        data: Input DataFrame
        model: Loaded VAE model
        scaler: Fitted scaler
        threshold_percentile: Percentile for anomaly threshold
    
    Returns:
        anomaly_scores: Reconstruction error per sample
        is_anomaly: Binary labels (1 = anomaly, 0 = normal)
    """
    # Prepare and predict
    features_scaled = scaler.transform(data)
    reconstructions = model.predict(features_scaled)
    
    # Calculate reconstruction error
    anomaly_scores = np.mean(np.square(features_scaled - reconstructions), axis=1)
    
    # Determine anomalies
    threshold = np.percentile(anomaly_scores, threshold_percentile)
    is_anomaly = (anomaly_scores > threshold).astype(int)
    
    return anomaly_scores, is_anomaly

# Example usage
anomaly_scores, is_anomaly = detect_anomalies_vae(data, vae_model, vae_scaler, threshold_percentile=95)

# Create results dataframe
results_df = pd.DataFrame({
    'sample_index': range(len(data)),
    'reconstruction_error': anomaly_scores,
    'is_anomaly': is_anomaly
})

print(results_df.head(10))
print(f"\nTotal anomalies: {is_anomaly.sum()}")
```

---

## Batch Prediction Workflow

### Process Multiple Batches

```python
def batch_predict(data, model, batch_size=32, model_type='lstm', scaler=None):
    """
    Process large datasets in batches to manage memory.
    
    Args:
        data: Input DataFrame
        model: Loaded model
        batch_size: Number of samples per batch
        model_type: 'lstm', 'svm', or 'vae'
        scaler: Fitted scaler
    
    Returns:
        all_predictions: Predictions for entire dataset
    """
    all_predictions = []
    
    for i in range(0, len(data), batch_size):
        batch = data.iloc[i:i + batch_size]
        
        if model_type == 'lstm':
            batch_sequences, _ = prepare_lstm_data(batch, scaler=scaler)
            predictions = model.predict(batch_sequences)
        elif model_type == 'svm':
            batch_features, _ = prepare_svm_data(batch, scaler=scaler)
            predictions = model.predict(batch_features)
        elif model_type == 'vae':
            batch_features, _ = prepare_vae_data(batch, scaler=scaler)
            predictions = model.predict(batch_features)
        
        all_predictions.append(predictions)
        print(f"Processed batch {i//batch_size + 1}")
    
    return np.concatenate(all_predictions, axis=0)

# Example: Batch prediction with SVM
svm_batch_predictions = batch_predict(data, svm_model, batch_size=64, 
                                      model_type='svm', scaler=svm_scaler)
print(f"Batch predictions shape: {svm_batch_predictions.shape}")
```

---

## Prediction Result Interpretation

```python
# Create comprehensive results dataframe
results_comparison = pd.DataFrame({
    'sample_id': range(len(data)),
    'lstm_prediction': predicted_classes if lstm_predictions.ndim > 1 else lstm_predictions.flatten(),
    'lstm_confidence': confidence_scores,
    'svm_prediction': svm_predictions,
    'vae_reconstruction_error': reconstruction_error,
    'vae_is_anomaly': anomaly_predictions
})

print(results_comparison.head(10))

# Class mapping (adjust based on your training labels)
class_mapping = {0: 'Normal', 1: 'Malfunction', 2: 'DoS Attack'}

results_comparison['lstm_class'] = results_comparison['lstm_prediction'].map(class_mapping)
results_comparison['svm_class'] = results_comparison['svm_prediction'].map(class_mapping)

print("\nResults with class labels:")
print(results_comparison[['sample_id', 'lstm_class', 'lstm_confidence', 'svm_class', 'vae_is_anomaly']].head(10))

# Save results
results_comparison.to_csv('predictions_results.csv', index=False)
print("\nResults saved to 'predictions_results.csv'")
```

---

# Troubleshooting

### 1. Notebook Kernel Issues
If cells fail to execute, restart the kernel and run from the beginning.

### 2. TensorFlow/CUDA Compatibility
Ensure GPU drivers and CUDA toolkit are compatible with TensorFlow 2.19.0. If issues persist:
```
pip install --upgrade tensorflow==2.19.0 tf-keras==2.19.0
```

### 3. Memory Issues
- Reduce LSTM batch size (default: 32, try: 16 or 8)
- Reduce sequence window size for temporal windowing
- Use CPU if GPU memory is limited

### 4. OpenCV Installation
If OpenCV fails to import, use:
```
pip install opencv-python-headless opencv-python opencv-contrib-python
```

### 5. Model Loading Errors
- Ensure joblib, tensorflow, and scikit-learn versions match those used during training
- For `.keras` files, use TensorFlow 2.19.0+
- For `.pkl` files, use matching scikit-learn versions

---

# Future Improvements

- **Real-Time API Deployment**: Deploy trained models as REST API for live drone monitoring
- **Additional Anomaly Detection**: Implement Isolation Forest, Deep SVDD, and One-Class SVM
- **Interactive Dashboards**: Create Streamlit/Dash dashboards for real-time visualization
- **Multi-Drone Support**: Extend pipeline for ingestion and analysis of multiple drone telemetry streams
- **Transfer Learning**: Pre-train on larger datasets and fine-tune for specific drone models
- **Edge Deployment**: Optimize models for deployment on embedded systems and edge devices
- **Confidence Intervals**: Add uncertainty quantification for production predictions
- **Continuous Learning**: Implement online learning pipeline for model updates with new data

---

# License
This project is licensed under the MIT License. See `LICENSE` for details.

---

# Contact
**Developer:** Muhammad Talha  
**Institute:** FAST - NUCES  
For queries, collaborations, or enterprise solutions:  
`muhammedtalha81@example.com`
