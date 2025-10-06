
# Cardiotoxicity Prediction Model - Usage Instructions

## Overview
This model predicts the risk of cancer treatment-related cardiac dysfunction (CTRCD) 
in HER2+ breast cancer patients undergoing cardiotoxic therapy.

**Model Version:** 1.0
**Test ROC AUC:** 0.7960
**Created:** October 2025

## Quick Start

### 1. Load the Model and Pipelines

```python
import pickle
import pandas as pd
import numpy as np

# Load model
with open('cardiotoxicity_model.pkl', 'rb') as f:
    model = pickle.load(f)

# Load pipelines
with open('preprocessing_pipeline.pkl', 'rb') as f:
    preprocessing = pickle.load(f)

with open('feature_engineering_pipeline.pkl', 'rb') as f:
    feature_engineering = pickle.load(f)

# Load metadata
import json
with open('model_metadata.json', 'r') as f:
    metadata = json.load(f)

selected_features = metadata['features']['selected_features']
```

### 2. Prepare Input Data

Your input data must contain the following 20 columns:

**Continuous Variables:**
- age (years)
- weight (kg)
- height (cm)
- heart_rate (bpm)
- LVEF (%)
- PWT (cm)
- LAd (cm)
- LVDd (cm)
- LVSd (cm)

**Binary Variables (0 or 1):**
- AC (anthracycline treatment)
- antiHER2 (anti-HER2 treatment)
- HTA (hypertension)
- DL (dyslipidemia)
- smoker (current smoker)
- exsmoker (former smoker)
- diabetes
- obesity
- ACprev (previous anthracycline)
- RTprev (previous radiotherapy)
- heart_rhythm (abnormal rhythm)

### 3. Make Predictions

```python
# Example patient data
patient_data = pd.DataFrame({
    'age': [55],
    'weight': [70],
    'height': [165],
    'heart_rate': [75],
    'LVEF': [60],
    'PWT': [0.9],
    'LAd': [3.5],
    'LVDd': [4.8],
    'LVSd': [3.2],
    'AC': [1],
    'antiHER2': [1],
    'HTA': [1],
    'DL': [0],
    'smoker': [0],
    'exsmoker': [0],
    'diabetes': [0],
    'obesity': [0],
    'ACprev': [0],
    'RTprev': [0],
    'heart_rhythm': [0]
})

# Step 1: Preprocess
processed_data = preprocessing.preprocess(patient_data)

# Step 2: Engineer features
engineered_data = feature_engineering.engineer_features(processed_data)

# Step 3: Select required features
X = engineered_data[selected_features]

# Step 4: Predict
risk_probability = model.predict_proba(X)[:, 1][0]
risk_class = model.predict(X)[0]

print(f"CTRCD Risk Probability: {risk_probability:.2%}")
print(f"Risk Class: {'High Risk' if risk_class == 1 else 'Low Risk'}")
```

### 4. Interpret Results

**Risk Thresholds:**
- **< 30%**: Low Risk - Standard monitoring
- **30-50%**: Moderate Risk - Enhanced monitoring
- **50-70%**: High Risk - Cardioprotective strategies
- **> 70%**: Very High Risk - Intensive monitoring and intervention

**Important Notes:**
- Default threshold is 0.5 (50%)
- Adjust threshold based on clinical context
- High sensitivity (threshold ~0.3): Fewer missed cases, more false alarms
- High specificity (threshold ~0.7): Fewer false alarms, may miss some cases

### 5. Clinical Interpretation

The model considers these key risk factors:

1. **Cardiovascular Risk Score**: Composite of age, CV risk factors, and baseline LVEF
2. **Heart Rate**: Elevated heart rate (especially cubic term) indicates cardiac stress
3. **Treatment Burden**: Cumulative cardiotoxic treatments (current + prior)
4. **Baseline Cardiac Function**: LVEF adjusted for age
5. **Risk-Treatment Interaction**: CV risk factors compound with treatment effects

## Batch Prediction

```python
# Load dataset with multiple patients
patients_df = pd.read_csv('patients.csv')

# Validate data
is_valid, message = preprocessing.validate(patients_df)
if not is_valid:
    print(f"Validation error: {message}")
else:
    # Process pipeline
    processed = preprocessing.preprocess(patients_df)
    engineered = feature_engineering.engineer_features(processed)
    X = engineered[selected_features]
    
    # Predict
    probabilities = model.predict_proba(X)[:, 1]
    
    # Add to results
    patients_df['ctrcd_risk'] = probabilities
    patients_df['risk_category'] = pd.cut(
        probabilities, 
        bins=[0, 0.3, 0.5, 0.7, 1.0],
        labels=['Low', 'Moderate', 'High', 'Very High']
    )
    
    # Save results
    patients_df.to_csv('predictions.csv', index=False)
```

## Feature Importance

Top 10 most important features:
1. CV_risk_score (17.03%)
2. heart_rate_cubed (13.87%)
3. risk_x_treatment (13.53%)
4. cumulative_cardiotoxic_treatment (9.40%)
5. age_adjusted_LVEF (7.51%)
6. heart_rate_squared (6.43%)
7. heart_rate (5.53%)
8. LVEF_x_n_risk_factors (3.10%)
9. age_cubed (2.63%)
10. LVEF (2.44%)

## Model Performance

- **Test ROC AUC:** 0.7960
- **Test PR AUC:** 0.4150
- **Improvement over baseline:** +30.62%

## Troubleshooting

**Missing Values:**
- Automatically handled by preprocessing pipeline
- Continuous: imputed with median from training data
- Binary: imputed with mode from training data

**Invalid Input:**
- Use `preprocessing.validate(df)` to check data before prediction
- Ensure all required columns are present
- Check that values are in valid ranges

**Dependencies:**
```bash
pip install pandas numpy scikit-learn xgboost==2.0.0
```

## Clinical Decision Support Workflow

1. **Input patient data** → Clinical measurements and treatment history
2. **Run prediction** → Get CTRCD risk probability
3. **Interpret risk** → Use thresholds and clinical context
4. **Clinical action:**
   - Low risk: Standard 3-6 month follow-up
   - Moderate risk: Enhanced monitoring (every 3 months)
   - High risk: Cardioprotective agents + frequent monitoring
   - Very high risk: Consider treatment modification + cardiology consult

## Contact & Support

For questions about model usage, interpretation, or deployment:
- Review model_metadata.json for technical details
- Consult cardiology team for clinical decision-making
- Monitor model performance in clinical practice

## License & Disclaimer

This model is for research and clinical decision support only. 
It should not be used as the sole basis for clinical decisions.
Always combine model predictions with clinical judgment and 
additional diagnostic information.

**Last Updated:** October 2025
**Model Version:** 1.0
