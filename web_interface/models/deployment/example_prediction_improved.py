#!/usr/bin/env python3
"""
Example: Cardiotoxicity Risk Prediction
Demonstrates how to use the deployed model for individual patient prediction
"""

import pandas as pd
from cardiotoxicity_predictor import CardiotoxicityPredictor

print("="*80)
print("CARDIOTOXICITY PREDICTION - EXAMPLE")
print("="*80)

# Initialize predictor
print("\nLoading model and pipelines...")
predictor = CardiotoxicityPredictor('.')
print("✓ Model loaded successfully!\n")

# Example patient scenarios
print("="*80)
print("EXAMPLE PATIENT PREDICTIONS")
print("="*80)

# Patient 1: Low risk profile
patient_1 = {
    'age': 45,
    'weight': 65,
    'height': 160,
    'heart_rate': 70,
    'LVEF': 65,  # Good cardiac function
    'PWT': 0.8,
    'LAd': 3.2,
    'LVDd': 4.5,
    'LVSd': 2.8,
    'AC': 0,  # No anthracycline
    'antiHER2': 1,
    'HTA': 0,  # No hypertension
    'DL': 0,
    'smoker': 0,
    'exsmoker': 0,
    'diabetes': 0,
    'obesity': 0,
    'ACprev': 0,
    'RTprev': 0,
    'heart_rhythm': 0
}

# Patient 2: High risk profile
patient_2 = {
    'age': 62,
    'weight': 80,
    'height': 165,
    'heart_rate': 88,  # Elevated heart rate
    'LVEF': 52,  # Borderline cardiac function
    'PWT': 1.1,
    'LAd': 4.2,
    'LVDd': 5.5,
    'LVSd': 3.8,
    'AC': 1,  # Receiving anthracycline
    'antiHER2': 1,  # Receiving anti-HER2
    'HTA': 1,  # Hypertension present
    'DL': 1,  # Dyslipidemia present
    'smoker': 0,
    'exsmoker': 1,
    'diabetes': 1,  # Diabetes present
    'obesity': 1,  # Obesity present
    'ACprev': 1,  # Previous anthracycline
    'RTprev': 1,  # Previous radiotherapy
    'heart_rhythm': 0
}

def print_patient_results(patient_dict, patient_name):
    """Print formatted prediction results for a patient"""
    print(f"\n{patient_name}:")
    print("-" * 80)
    
    # Get prediction
    result = predictor.predict_single(patient_dict)
    
    print(f"  CTRCD Risk Probability: {result['probability']:.2%}")
    print(f"  Risk Level: {result['risk_level']}")
    print(f"  Clinical Recommendation: {result['recommendation']}")
    
    # Show key clinical parameters
    print(f"\n  Key Clinical Parameters:")
    print(f"    - Age: {patient_dict['age']} years")
    print(f"    - LVEF: {patient_dict['LVEF']}%")
    print(f"    - Heart Rate: {patient_dict['heart_rate']} bpm")
    
    cv_risk_count = sum([
        patient_dict['HTA'],
        patient_dict['DL'],
        patient_dict['smoker'] or patient_dict['exsmoker'],
        patient_dict['diabetes'],
        patient_dict['obesity']
    ])
    print(f"    - CV Risk Factors: {cv_risk_count}")
    
    treatment_burden = sum([
        patient_dict['AC'],
        patient_dict['antiHER2'],
        patient_dict['ACprev'],
        patient_dict['RTprev']
    ])
    print(f"    - Treatment Burden: {treatment_burden}")

# Run predictions
print_patient_results(patient_1, "PATIENT 1: Low Risk Profile")
print_patient_results(patient_2, "PATIENT 2: High Risk Profile")

print("\n" + "="*80)

# Batch prediction example
print("\nBATCH PREDICTION EXAMPLE")
print("="*80)

# Create a small batch of patients
batch_patients = pd.DataFrame([patient_1, patient_2])
batch_results = predictor.predict(batch_patients)

print(f"\nProcessed {len(batch_patients)} patients:")
for i, (prob, level, rec) in enumerate(zip(
    batch_results['probabilities'], 
    batch_results['risk_levels'],
    batch_results['recommendations']
), 1):
    print(f"\n  Patient {i}:")
    print(f"    Risk: {prob:.2%} ({level})")
    print(f"    Action: {rec}")

print("\n" + "="*80)
print("Predictions completed successfully!")
print("="*80)
