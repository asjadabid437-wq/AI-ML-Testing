# ✈️ Airline Passenger Satisfaction Prediction System

### End-to-End Machine Learning | Client Project

## 📌 Project Overview

The Airline Passenger Satisfaction Prediction System is an end-to-end Machine Learning project developed to solve a real-world business problem for a client.

The objective of this project is to analyze airline passenger experiences and predict whether passengers are satisfied or neutral/dissatisfied based on various travel-related and service-related factors.

I independently handled the complete project development lifecycle, including data analysis, preprocessing, feature engineering, Machine Learning model development, evaluation, and prediction system implementation.

## 💼 Client Project Background

A client required an intelligent solution to analyze airline passenger satisfaction and predict customer experience using Machine Learning.

To address this requirement, I designed and developed a predictive system that analyzes passenger-related information and generates satisfaction predictions.

### Client Requirements

- Develop a Machine Learning-based prediction system.
- Analyze airline passenger satisfaction.
- Identify important factors affecting customer experience.
- Predict passenger satisfaction automatically.
- Provide an easy-to-use prediction interface.

## 🎯 Project Objectives

- Build an intelligent passenger satisfaction prediction system.
- Perform detailed data analysis.
- Apply appropriate Machine Learning techniques.
- Evaluate model performance.
- Generate passenger satisfaction predictions.
- Support data-driven business decisions.

## 🛠️ Technologies & Tools

| Technology | Purpose |
|---|---|
| Python | Main Programming Language |
| Pandas | Data Analysis |
| NumPy | Numerical Operations |
| Matplotlib | Data Visualization |
| Seaborn | Statistical Visualization |
| Scikit-learn | Machine Learning |
| Flask | Web Application (if used) |
| HTML & CSS | Frontend (if used) |

## ⚙️ Key Features

- Data Loading and Exploration
- Data Cleaning and Preprocessing
- Missing Value Handling
- Feature Engineering
- Categorical Data Transformation
- Machine Learning Model Training
- Model Evaluation
- Passenger Satisfaction Prediction
- Web-Based Prediction Interface

## 🔄 Machine Learning Workflow

1. **Data Understanding:** Analyzing dataset structure, features, and target variables.
2. **Exploratory Data Analysis (EDA):** Understanding patterns, relationships, and important factors.
3. **Data Preprocessing:** Cleaning data and preparing it for Machine Learning.
4. **Feature Engineering:** Transforming categorical and numerical features.
5. **Model Development:** Building and training Machine Learning models.
6. **Model Evaluation:** Evaluating performance using classification metrics.
7. **Prediction System:** Predicting passenger satisfaction based on input information.

## 📈 Exploratory Data Analysis (EDA) & Insights

### Passenger Gender Distribution
<img width="693" height="572" alt="Screenshot 2026-11-22 134421" src="https://github.com/user-attachments/assets/9ba016e8-a6fe-4244-ad8e-b0676fe41901" />

### Customer Type Distribution
![Customer Type Distribution](screenshots/customer-type-distribution.png)

### Satisfaction by Travel Class
![Satisfaction by Travel Class](screenshots/satisfaction-travel-class.png)

### Passenger Age Distribution
![Passenger Age Distribution](screenshots/age-distribution.png)

### Feature Correlation Heatmap
![Feature Correlation Heatmap](screenshots/correlation-heatmap.png)

### Top 15 Passenger Satisfaction Features
![Top 15 Passenger Satisfaction Features](screenshots/feature-importance.png)

## 📊 Model Evaluation & Results

The models (**Logistic Regression**, **Decision Tree**, and **Random Forest**) are evaluated using:

- Accuracy
- Precision
- Recall
- F1-Score
- ROC-AUC Score
- Confusion Matrix

### Best Model Performance Summary (Random Forest)

| Metric | Score |
|---|---|
| **Test Accuracy** | **96.32%** (0.9632) |
| **Test Precision** | **97.20%** (0.9720) |
| **Test Recall** | **94.33%** (0.9433) |
| **Test F1 Score** | **95.75%** (0.9575) |
| **Test ROC AUC** | **99.38%** (0.9938) |

### Machine Learning Model Comparison
![Machine Learning Model Comparison](screenshots/model-comparison.png)

### Final Model Confusion Matrix
![Final Model Confusion Matrix](screenshots/confusion-matrix.png)

### Terminal Execution & Classification Report
![Final Testing Results](screenshots/final-testing-results.png)
![Classification Report](screenshots/classification-report.png)

## 📸 Application Screenshots

### Home Page
![Home Page](screenshots/home.png)

### Prediction Interface
![Prediction Interface](screenshots/prediction.png)

### Prediction Results
![Prediction Results](screenshots/result.png)

## 📁 Project Structure

```text
Passenger_Satisfaction_AI/
│
├── dataset/
│   ├── train.csv
│   └── test.csv
│
├── screenshots/
│   ├── home.png
│   ├── prediction.png
│   ├── result.png
│   ├── gender-distribution.png
│   ├── customer-type-distribution.png
│   ├── satisfaction-travel-class.png
│   ├── age-distribution.png
│   ├── correlation-heatmap.png
│   ├── feature-importance.png
│   ├── model-comparison.png
│   ├── confusion-matrix.png
│   ├── final-testing-results.png
│   └── classification-report.png
│
├── .vscode/
│
├── app.py
├── main.py
├── predict.py
├── best_passenger_model.pkl
├── feature_importance.csv
├── model_comparison_results.csv
├── requirements.txt
└── README.md
