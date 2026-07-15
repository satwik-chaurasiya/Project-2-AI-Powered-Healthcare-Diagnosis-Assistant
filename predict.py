import pandas as pd
import numpy as np
import joblib

def evaluate_patient(input_data):
    """
    input_data: dict containing age, gender, weight, height, temperature, symptoms (list)
    """
    # Load Models & Configuration
    model = joblib.load('model/disease_model.pkl')
    scaler = joblib.load('model/scaler.pkl')
    symptom_cols = joblib.load('model/symptom_encoder.pkl')
    
    severity_df = pd.read_csv('dataset/severity.csv').set_index('symptom')
    weights = severity_df['weight'].to_dict()
    
    # Module 11 — Emergency Detection
    symptoms_set = set(input_data['symptoms'])
    if 'chest_pain' in symptoms_set and 'breathing_difficulty' in symptoms_set:
        return {"emergency": True, "message": "CRITICAL WARNING: Chest pain combined with breathing difficulty detected. Seek emergency medical care immediately!"}
    
    # Preprocessing & Vectorization (Module 2 & 4)
    symptom_vector = {sym: (1 if sym in symptoms_set else 0) for sym in symptom_cols}
    
    bmi = float(input_data['weight']) / (float(input_data['height']) ** 2)
    symptom_count = sum(symptom_vector.values())
    severity_score = sum(symptom_vector[sym] * weights[sym] for sym in symptom_cols)
    
    # Combine engineered array
    base_features = [
        float(input_data['age']),
        bmi,
        float(input_data['temperature']),
        symptom_count,
        severity_score
    ]
    
    final_features = [symptom_vector[sym] for sym in symptom_cols] + base_features
    final_features_scaled = scaler.transform([final_features])
    
    # Module 6 — Confidence Score (Top-N Predictions)
    probabilities = model.predict_proba(final_features_scaled)[0]
    classes = model.classes_
    
    top_indices = np.argsort(probabilities)[::-1][:3]
    predictions = [{"disease": classes[i], "probability": round(probabilities[i] * 100, 2)} for i in top_indices]
    primary_disease = predictions[0]['disease']
    
    # Module 7 — Risk Prediction scoring engine
    risk_score = 0
    if float(input_data['age']) >= 65: risk_score += 20
    if float(input_data['temperature']) > 39.0: risk_score += 25
    if 'breathing_difficulty' in symptoms_set: risk_score += 40
    risk_score += (severity_score * 2)
    
    if risk_score <= 30:
        risk_level = "Low"
    elif risk_score <= 60:
        risk_level = "Medium"
    else:
        risk_level = "High"
        
    # Module 8, 9, 10 — Metadata resolution
    doc_df = pd.read_csv('dataset/doctors.csv').set_index('disease')
    med_df = pd.read_csv('dataset/medications.csv').set_index('disease')
    prec_df = pd.read_csv('dataset/precautions.csv').set_index('disease')
    
    specialist = doc_df.loc[primary_disease, 'specialist'] if primary_disease in doc_df.index else "General Physician"
    medicine = med_df.loc[primary_disease, 'medicine'] if primary_disease in med_df.index else "Consult a Doctor"
    
    prec_row = prec_df.loc[primary_disease] if primary_disease in prec_df.index else None
    precautions = [prec_row['precaution_1'], prec_row['precaution_2']] if prec_row is not None else ["Rest", "Consult specialist"]
    
    return {
        "emergency": False,
        "predictions": predictions,
        "risk_score": risk_score,
        "risk_level": risk_level,
        "specialist": specialist,
        "medicine_info": medicine,
        "precautions": precautions,
        "bmi": round(bmi, 1)
    }