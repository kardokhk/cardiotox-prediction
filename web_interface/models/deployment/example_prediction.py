#!/usr/bin/env python3
"""
Example: Cardiotoxicity Risk Prediction
Demonstrates how to use the deployed model for individual patient prediction
"""

import pickle
import pandas as pd
import json

# Load model and pipelines
print("Loading model and pipelines...")
with open('cardiotoxicity_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('preprocessing_pipeline.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

with open('feature_engineering_pipeline.pkl', 'rb') as f:
    feature_engineering = pickle.load(f)

with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

selected_features = metadata['features']['selected_features']
print("✓ Model loaded successfully!\n")

# Example patient scenarios
print("="*80)
print("EXAMPLE PATIENT PREDICTIONS")
print("="*80)

# Patient 1: Low risk profile
patient_1 = pd.DataFrame({
    'age': [45],
    'weight': [65],
    'height': [160],
    'heart_rate': [70],
    'LVEF': [65],  # Good cardiac function
    'PWT': [0.8],
    'LAd': [3.2],
    'LVDd': [4.5],
    'LVSd': [2.8],
    'AC': [0],  # No anthracycline
    'antiHER2': [1],
    'HTA': [0],  # No hypertension
    'DL': [0],
    'smoker': [0],
    'exsmoker': [0],
    'diabetes': [0],
    'obesity': [0],
    'ACprev': [0],
    'RTprev': [0],
    'heart_rhythm': [0]
})

# Patient 2: High risk profile
patient_2 = pd.DataFrame({
    'age': [62],
    'weight': [80],
    'height': [165],
    'heart_rate': [88],  # Elevated heart rate
    'LVEF': [52],  # Borderline cardiac function
    'PWT': [1.1],
    'LAd': [4.2],
    'LVDd': [5.5],
    'LVSd': [3.8],
    'AC': [1],  # Receiving anthracycline
    'antiHER2': [1],  # Receiving anti-HER2
    'HTA': [1],  # Hypertension present
    'DL': [1],  # Dyslipidemia present
    'smoker': [0],
    'exsmoker': [1],
    'diabetes': [1],  # Diabetes present
    'obesity': [1],  # Obesity present
    'ACprev': [1],  # Previous anthracycline
    'RTprev': [1],  # Previous radiotherapy
    'heart_rhythm': [0]
})

def predict_patient(patient_data, patient_name):
    """Predict CTRCD risk for a patient"""
    print(f"\n{patient_name}:")
    print("-" * 80)
    
    # Validate
    is_valid, message = preprocessing.validate(patient_data)
    if not is_valid:
        print(f"❌ Validation failed: {message}")
        return
    
    # Process
    processed = preprocessing.preprocess(patient_data)
    engineered = feature_engineering.engineer_features(processed)
    X = engineered[selected_features]
    
    # Predict
    risk_prob = model.predict_proba(X)[:, 1][0]
    
    # Interpret
    if risk_prob < 0.3:
        risk_level = "LOW"
        recommendation = "Standard monitoring (every 6 months)"
    elif risk_prob < 0.5:
        risk_level = "MODERATE"
        recommendation = "Enhanced monitoring (every 3 months)"
    elif risk_prob < 0.7:
        risk_level = "HIGH"
        recommendation = "Cardioprotective agents + frequent monitoring"
    else:
        risk_level = "VERY HIGH"
        recommendation = "Consider treatment modification + cardiology consult"
    
    print(f"  CTRCD Risk Probability: {risk_prob:.2%}")
    print(f"  Risk Level: {risk_level}")
    print(f"  Clinical Recommendation: {recommendation}")
    
    # Show key risk factors
    print(f"\n  Key Clinical Parameters:")
    print(f"    - Age: {patient_data['age'].iloc[0]} years")
    print(f"    - LVEF: {patient_data['LVEF'].iloc[0]}%")
    print(f"    - Heart Rate: {patient_data['heart_rate'].iloc[0]} bpm")
    
    cv_risk_count = sum([
        patient_data['HTA'].iloc[0],
        patient_data['DL'].iloc[0],
        patient_data['smoker'].iloc[0] or patient_data['exsmoker'].iloc[0],
        patient_data['diabetes'].iloc[0],
        patient_data['obesity'].iloc[0]
    ])
    print(f"    - CV Risk Factors: {cv_risk_count}")
    
    treatment_burden = sum([
        patient_data['AC'].iloc[0],
        patient_data['antiHER2'].iloc[0],
        patient_data['ACprev'].iloc[0],
        patient_data['RTprev'].iloc[0]
    ])
    print(f"    - Treatment Burden: {treatment_burden}")

# Run predictions
predict_patient(patient_1, "PATIENT 1: Low Risk Profile")
predict_patient(patient_2, "PATIENT 2: High Risk Profile")

print("\n" + "="*80)
print("Predictions completed successfully!")
print("="*80)
