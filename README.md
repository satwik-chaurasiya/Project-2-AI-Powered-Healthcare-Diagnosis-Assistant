---
## AI-Powered Healthcare Diagnosis Assistant 🩺✨

An end-to-end Machine Learning web application that evaluates patient demographics, vitals, and selected symptoms to estimate risk tiers, output predictive diagnostic matches, and seamlessly route users to specialized healthcare professionals. 

> ⚠️ **CRITICAL DISCLAIMER:** This platform serves purely informational and educational metrics. It does not provide definitive medical diagnoses, clinical evaluations, or active prescriptions. Always consult a certified healthcare professional for medical advice.

---

## 🧱 Project Directory Structure

```text
AI_Healthcare_Diagnosis_Assistant/
│
├── dataset/
│   ├── symptoms.csv
│   ├── diseases.csv
│   ├── precautions.csv
│   ├── medications.csv
│   ├── doctors.csv
│   └── severity.csv
│
├── model/
│   ├── disease_model.pkl
│   ├── symptom_encoder.pkl
│   └── scaler.pkl
│
├── templates/
│   ├── index.html
│   ├── prediction.html
│   └── history.html
│
├── app.py
├── train_model.py
├── predict.py
├── database.py
├── requirements.txt
└── README.md
```
---

## ⚙️ Core Modules & Features

* **Module 1: Patient Matrix Collection:** Gathers profile data including Age, Gender, Weight, Height, Temperature, and symptom checklists.
* **Module 2 & 4: Feature Engineering & Encoding:** Translates dynamic textual symptoms into structured multi-hot binary vectors. Synthesizes vital data metrics such as Body Mass Index ($BMI = \frac{\text{Weight}}{\text{Height}^2}$) and customized symptom severity indices.
* **Module 5 & 6: Machine Learning Pipeline:** Employs a robust Random Forest Classifier to infer statistical Top-3 disease probabilities ($N$-class predictions).
* **Module 7: Algorithmic Triage Engine:** Automatically computes a composite risk index to map patient severity into Low, Medium, or High tracking profiles.
* **Module 8, 9 & 10: Rule-Based Recommendations:** References isolated internal mapping layers to present non-prescriptive, informational care alternatives and map recommended clinical specialists.
* **Module 11: Emergency Triage Vectoring:** Scans symptoms instantly for critical conditions (e.g., Chest Pain + Breathing Difficulty) to intercept processing and issue an immediate emergency warning.
* **Module 13: Local Historical Ledger:** Implements a locally containerized SQLite database layer to maintain records of patient evaluation logs securely.

---

## 🛡️ Ethical, Safety, & Data Privacy Commitments

* **Clinical Boundary Separation:** System text outputs explicitly state "Likely Conditions" instead of definitive medical diagnostic statements.
* **Emergency Overrides:** Rule configurations completely bypass standard model estimation flows if symptoms show life-threatening anomalies.
* **Local Processing Boundaries:** All data transformations, feature metrics, and database transactions occur natively on your host machine to secure patient anonymity.

---

## 🧰 Technology Stack Matrix

| Layer | Component | Description |
| :--- | :--- | :--- |
| **Frontend** | HTML5, CSS3, Bootstrap 5 | Mobile-first responsive user interface dashboard |
| **Backend** | Flask | Central processing server core control router |
| **Security** | Flask-WTF (CSRFProtect) | Mitigates Cross-Site Request Forgery threats globally |
| **Data Processing** | Pandas, NumPy | Structured matrix transformations and arithmetic pipelines |
| **Machine Learning** | Scikit-learn | Classification algorithms and pipeline transformations |
| **Storage** | SQLite3 | Historical clinical logging ledger |
| **Serialization** | Joblib | Compresses and saves trained pipeline states |
