import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report

# # 1. Load the new binary diabetes dataset
try:
    df = pd.read_csv(r"C:\Users\satwi\OneDrive\Desktop\diabetes_binary_5050split_health_indicators_BRFSS2015.csv")
    df = df.dropna(how='all')
except pd.errors.EmptyDataError:
    df = pd.DataFrame() 

if df.empty:
    print("⚠️ Warning: The dataset file is currently empty or missing columns. Please populate it with patient data.")
else:
    # # 2. Separate features and target
    X = df.drop(columns=['Diabetes_binary'])
    y = df['Diabetes_binary']

    # # 3. Split into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

    # # 4. Scale medical features for uniformity
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # # # 5. Train a robust Classifier
    model = RandomForestClassifier(
        n_estimators=100,
        min_samples_leaf=1,
        max_features='sqrt',
        random_state=42
    )
    model.fit(X_train_scaled, y_train)

    # # # 6. Evaluate the Assistant's Accuracy
    predictions = model.predict(X_test_scaled)
    print(f"Model Diagnostics Accuracy: {accuracy_score(y_test, predictions) * 100:.2f}%")
    print("\nDetailed Clinical Report:\n", classification_report(y_test, predictions))

    # Example function for clinical deployment
    def assess_new_patient(patient_metrics):
        scaled_metrics = scaler.transform([patient_metrics])
        risk_probability = model.predict_proba(scaled_metrics)[0][1]
        return f"Risk of Diabetes condition: {risk_probability * 100:.1f}%"

    # --- NEW INTERACTIVE FEATURE FOR TERMINAL INPUT ---
    print("\n" + "="*40)
    print("📋 ENTER NEW PATIENT DATA FOR CLINICAL ASSESSMENT")
    print("="*40)
    
    new_patient_data = []
    
    # Dynamically loop through the feature columns of your new dataset
    for feature in X.columns:
        while True:
            try:
                val = float(input(f"Enter value for {feature}: "))
                new_patient_data.append(val)
                break
            except ValueError:
                print("❌ Invalid input. Please enter a numerical value.")
                
    print("\n" + "-"*40)
    print("Calculating diagnostic risk prediction...")
    result = assess_new_patient(new_patient_data)
    print(result)
    print("-"*40)