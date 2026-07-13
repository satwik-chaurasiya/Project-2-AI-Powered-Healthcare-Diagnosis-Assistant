# 📋 Interactive Diabetes Risk Prediction System

An end-to-end machine learning pipeline built to assist in the early diagnostic screening of diabetes risk using behavioral, demographic, and physical health datasets (`diabetes_binary_5050split_health_indicators_BRFSS2015.csv`). This project provides a production-compliant classification system coupled with a real-time terminal deployment interface for clinical assessment simulation.

---

## 🎯 Project Objective
The primary objective of this project is to build a highly sensitive predictive tool that evaluates a patient's risk of diabetes based on everyday clinical and behavioral indicators. By optimizing for strong positive class classification metrics, the system minimizes false negatives, ensuring individuals with underlying high-risk factors can be flagged early for preventative care.

---

## ⚙️ Core System Features
* **21-Factor Comprehensive Analysis:** Evaluates a wide array of indicators including blood pressure (`HighBP`), cholesterol status (`HighChol`, `CholCheck`), body mass index (`BMI`), lifestyle habits (`Smoker`, `PhysActivity`, `Fruits`, `Veggies`), and demographic factors (`Age`, `Education`, `Income`).
* **Interactive Clinical Terminal:** Includes a safe, real-time command-line interface that loops through data columns dynamically, validates user input values against runtime execution errors, and computes localized risk instantly.
* **Production-Grade Architecture:** Fully optimized following rigorous SonarQube formatting and code quality metrics (including explicit hyperparameter profiles and string literal structural safety).

---

## 📊 Model Performance & Diagnostics
The pipeline utilizes an optimized **RandomForestClassifier** trained on balanced behavioral indicators:
* **Overall Diagnostics Accuracy:** **73.31%**
* **High Sensitivity (Recall: 0.77):** Specifically tailored for the positive diabetes class to prevent high-risk profiles from bypassing initial screening flags.
* **Stable Generalization:** Balanced macro and weighted scoring averages ensuring consistent performance across both non-diabetic and diabetic groups.

---

## 🛠️ Data Pipeline & Methodology
1. **Data Ingestion & Cleaning:** Safely parses targeted CSV inputs, stripping out empty space padding or unpopulated template artifacts to maintain zero-bias matrix inputs.
2. **Feature Uniformity:** Implements standard normal feature scaling via a `StandardScaler` pipeline to eliminate skewness introduced by different measurement scales (e.g., categorical metrics vs. numerical `BMI` values).
3. **Data Splitting:** Uses a stratified train-test split (80/20) to guarantee that distribution balances of the target attribute are perfectly preserved between model learning and test validation loops.

---

## 🚀 Method to Use
---

Ensure you have Python installed along with the required dependencies:
```text
│
├──pip install pandas numpy scikit-learn
│
├──Make sure that you have the csv file in your PC add add the path in:
└── df = pd.read_csv("YOUR_PATH_TO/diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
```
---