import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
import joblib

# Create structured directories
os.makedirs('dataset', exist_ok=True)
os.makedirs('model', exist_ok=True)

def generate_mock_data():
    # 1. Symptoms & Disease Map
    symptoms = ['fever', 'cough', 'headache', 'fatigue', 'vomiting', 'chest_pain', 'breathing_difficulty', 'rash']
    diseases = ['Flu', 'Food Poisoning', 'Cold', 'COVID-19', 'Migraine', 'Heart Attack', 'Skin Infection']
    
    rng = np.random.default_rng(42)
    data = []
    for _ in range(1000):
        row = {sym: int(rng.choice([0, 1], p=[0.7, 0.3])) for sym in symptoms}
        # Add engineered base metrics
        row['age'] = int(rng.integers(18, 80))
        row['weight'] = float(rng.uniform(50, 100))
        row['height'] = float(rng.uniform(1.5, 2.0))
        row['temperature'] = float(rng.uniform(36.0, 40.0))
        
        # Simple logical rules for assigning target labels to mock data
        if row['chest_pain'] == 1 and row['breathing_difficulty'] == 1:
            row['disease'] = 'Heart Attack'
        elif row['fever'] == 1 and row['cough'] == 1 and row['breathing_difficulty'] == 1:
            row['disease'] = 'COVID-19'
        elif row['vomiting'] == 1 and row['fever'] == 1:
            row['disease'] = 'Food Poisoning'
        elif row['headache'] == 1 and row['fatigue'] == 1 and row['fever'] == 0:
            row['disease'] = 'Migraine'
        elif row['rash'] == 1:
            row['disease'] = 'Skin Infection'
        elif row['fever'] == 1 or row['cough'] == 1:
            row['disease'] = 'Flu'
        else:
            row['disease'] = 'Cold'
            
        data.append(row)
        
    df = pd.DataFrame(data)
    df.to_csv('dataset/diseases.csv', index=False)

    # 2. Metadata reference files
    precautions = pd.DataFrame({
        'disease': diseases,
        'precaution_1': ['Hydration', 'Fluid replacement', 'Rest', 'Isolation', 'Dark room rest', 'Call Emergency', 'Keep clean'],
        'precaution_2': ['Rest', 'Avoid solid food', 'Warm fluids', 'Monitor oxygen', 'Avoid bright screens', 'Chew Aspirin', 'Apply topical gel']
    })
    precautions.to_csv('dataset/precautions.csv', index=False)

    doctors = pd.DataFrame({
        'disease': diseases,
        'specialist': ['General Physician', 'Gastroenterologist', 'General Physician', 'Pulmonologist', 'Neurologist', 'Cardiologist', 'Dermatologist']
    })
    doctors.to_csv('dataset/doctors.csv', index=False)

    medications = pd.DataFrame({
        'disease': diseases,
        'medicine': ['Paracetamol', 'OR Solution', 'Cetirizine', 'Remdesivir/Zinc', 'Sumatriptan', 'Nitroglycerin', 'Antihistamine']
    })
    medications.to_csv('dataset/medications.csv', index=False)

    severity = pd.DataFrame({
        'symptom': symptoms,
        'weight': [3, 2, 2, 2, 3, 5, 5, 2]
    })
    severity.to_csv('dataset/severity.csv', index=False)

def train():
    df = pd.read_csv('dataset/diseases.csv')
    severity_df = pd.read_csv('dataset/severity.csv').set_index('symptom')
    symptom_cols = list(severity_df.index)
    
    # Feature Engineering (Module 4)
    df['bmi'] = df['weight'] / (df['height'] ** 2)
    df['symptom_count'] = df[symptom_cols].sum(axis=1)
    
    # Calculate Severity Score
    weights = severity_df['weight'].to_dict()
    df['severity_score'] = df[symptom_cols].apply(lambda r: sum(r[sym] * weights[sym] for sym in symptom_cols), axis=1)
    
    # Define complete feature space
    features = symptom_cols + ['age', 'bmi', 'temperature', 'symptom_count', 'severity_score']
    X = df[features]
    y = df['disease']
    
    X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
    
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(X_train)
    
    model = RandomForestClassifier(
                                n_estimators=100,
                                max_features='sqrt',
                                min_samples_leaf=1,
                                random_state=42,
                                n_jobs=-1)
    model.fit(x_train_scaled, y_train)
    
    # Persist objects
    joblib.dump(model, 'model/disease_model.pkl')
    joblib.dump(scaler, 'model/scaler.pkl')
    joblib.dump(symptom_cols, 'model/symptom_encoder.pkl')
    print("Training finished successfully. Models saved to /model/")

if __name__ == '__main__':
    generate_mock_data()
    train()