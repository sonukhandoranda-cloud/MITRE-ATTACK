# MITRE ATT&CK Threat Group Classification using Machine Learning

## Overview

This project classifies cyber threat groups using the *MITRE ATT&CK STIX BUNDLE * dataset and Machine Learning techniques. The pipeline extracts behavioral features from ATT&CK techniques, handles class imbalance, trains multiple classifiers, performs hyperparameter optimization, and predicts the most likely threat groups.

A Streamlit dashboard is also provided for interactive threat group prediction.

---

## Features

- Parse MITRE ATT&CK STIX 2.1 dataset
- Feature Engineering
- Handle imbalanced data using SMOTE and Class Weights
- Multiple Machine Learning models
  - XGBoost
  - Random Forest
  - Support Vector Machine (SVM)
- Hyperparameter tuning using Optuna and GridSearchCV
- Model evaluation using Precision, Recall, F1-Score, Confusion Matrix, and SHAP Feature Importance
- Model serialization using Joblib
- Interactive Streamlit Dashboard
- Top-3 ranked predictions with confidence scores

## 📸 Dashboard

![Dashboard](Screenshot%202026-07-09%20203604.png)

![Dashboard](Screenshot%202026-07-09%20203621.png)
![Dashboard](Screenshot%202026-07-09%20202356.png)

![Dashboard](Screenshot%202026-07-09%20202413.png)



## Tech Stack

- Python 3.11
- Scikit-learn
- XGBoost
- Pandas
- NumPy
- SciPy
- Imbalanced-learn (SMOTE)
- Optuna
- SHAP
- Joblib
- Streamlit
- Git

---

## Dataset

- MITRE ATT&CK STIX 2.1 Dataset
- 41 STIX 2.1 JSON Files
- Behavioral records of cyber threat groups and ATT&CK techniques

---

## Machine Learning Pipeline

1. Data Ingestion
2. STIX Parsing
3. Feature Engineering
4. Matrix Serialization
5. Data Balancing (SMOTE/Class Weight)
6. Model Training
7. Hyperparameter Optimization
8. Model Evaluation
9. Model Serialization
10. Threat Group Prediction

---

## Evaluation Metrics

- Accuracy
- Precision
- Recall
- F1-Score
- Confusion Matrix
- SHAP Feature Importance

---

## Installation

bash
git clone https://github.com/yourusername/mitre-threat-classification.git

cd mitre-threat-classification

pip install -r requirements.txt

streamlit run app.py
