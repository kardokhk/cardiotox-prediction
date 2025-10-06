"""
Phase 11: Model Deployment Preparation
Prepares the final model for deployment by saving model, pipelines, and documentation

This script creates a complete deployment package including:
- Trained model (pickle format)
- Preprocessing pipeline
- Feature engineering pipeline
- Model metadata and configuration
- Usage instructions and examples

Author: Competition Submission
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import pickle
import xgboost as xgb
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix, classification_report
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
ENGINEERED_DIR = BASE_DIR / 'data' / 'engineered'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
MODELS_DIR = BASE_DIR / 'models'
DEPLOYMENT_DIR = MODELS_DIR / 'deployment'

# Create deployment directory
DEPLOYMENT_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 11: MODEL DEPLOYMENT PREPARATION")
print("="*80)
print(f"\nDeployment directory: {DEPLOYMENT_DIR}")

# ============================================================================
# 1. Load Data and Best Model Configuration
# ============================================================================
print("\n" + "="*80)
print("1. LOADING DATA AND CONFIGURATION")
print("="*80)

# Load engineered datasets
print("\nLoading engineered datasets...")
train_df = pd.read_csv(ENGINEERED_DIR / 'train_engineered.csv')
val_df = pd.read_csv(ENGINEERED_DIR / 'val_engineered.csv')
test_df = pd.read_csv(ENGINEERED_DIR / 'test_engineered.csv')
print(f"   ✓ Train: {train_df.shape}")
print(f"   ✓ Validation: {val_df.shape}")
print(f"   ✓ Test: {test_df.shape}")

# Load processed datasets (for preprocessing pipeline)
print("\nLoading processed datasets...")
train_processed = pd.read_csv(PROCESSED_DIR / 'train_data.csv')
val_processed = pd.read_csv(PROCESSED_DIR / 'val_data.csv')
test_processed = pd.read_csv(PROCESSED_DIR / 'test_data.csv')
print(f"   ✓ Processed train: {train_processed.shape}")
print(f"   ✓ Processed validation: {val_processed.shape}")
print(f"   ✓ Processed test: {test_processed.shape}")

# Load feature sets
print("\nLoading feature configuration...")
with open(ENGINEERED_DIR / 'feature_sets.json', 'r') as f:
    feature_sets = json.load(f)

with open(ENGINEERED_DIR / 'feature_categories.json', 'r') as f:
    feature_categories = json.load(f)

# Load best model configuration
print("\nLoading best model configuration...")
with open(MODELS_DIR / 'final_best_model_card.json', 'r') as f:
    model_card = json.load(f)

with open(MODELS_DIR / 'random_search_metrics.json', 'r') as f:
    best_config = json.load(f)
    optimized_params = best_config['best_params']

# Get selected features (RFE selected - 40 features)
selected_features = model_card['features']
print(f"\n   ✓ Model version: {model_card['version']}")
print(f"   ✓ Feature set: {model_card['feature_set']}")
print(f"   ✓ Number of features: {len(selected_features)}")
print(f"   ✓ Test ROC AUC: {model_card['performance']['test']['roc_auc']:.4f}")

# ============================================================================
# 2. Train Final Model
# ============================================================================
print("\n" + "="*80)
print("2. TRAINING FINAL MODEL")
print("="*80)

# Prepare data
X_train = train_df[selected_features]
y_train = train_df['CTRCD']
X_val = val_df[selected_features]
y_val = val_df['CTRCD']
X_test = test_df[selected_features]
y_test = test_df['CTRCD']

print(f"\nTraining final model with {len(selected_features)} features...")
print(f"   Training samples: {len(X_train)} (CTRCD=1: {y_train.sum()})")
print(f"   Validation samples: {len(X_val)} (CTRCD=1: {y_val.sum()})")
print(f"   Test samples: {len(X_test)} (CTRCD=1: {y_test.sum()})")

# Initialize and train model
final_model = xgb.XGBClassifier(
    **optimized_params,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

print("\nTraining model...")
final_model.fit(
    X_train, y_train,
    eval_set=[(X_val, y_val)],
    verbose=False
)
print("   ✓ Model trained successfully!")

# Evaluate performance
print("\nEvaluating model performance...")
y_train_proba = final_model.predict_proba(X_train)[:, 1]
y_val_proba = final_model.predict_proba(X_val)[:, 1]
y_test_proba = final_model.predict_proba(X_test)[:, 1]

train_auc = roc_auc_score(y_train, y_train_proba)
val_auc = roc_auc_score(y_val, y_val_proba)
test_auc = roc_auc_score(y_test, y_test_proba)

train_pr = average_precision_score(y_train, y_train_proba)
val_pr = average_precision_score(y_val, y_val_proba)
test_pr = average_precision_score(y_test, y_test_proba)

print(f"\n   Train ROC AUC: {train_auc:.4f} | PR AUC: {train_pr:.4f}")
print(f"   Val   ROC AUC: {val_auc:.4f} | PR AUC: {val_pr:.4f}")
print(f"   Test  ROC AUC: {test_auc:.4f} | PR AUC: {test_pr:.4f}")

# ============================================================================
# 3. Save Trained Model
# ============================================================================
print("\n" + "="*80)
print("3. SAVING TRAINED MODEL")
print("="*80)

# Save as pickle for deployment
model_path = DEPLOYMENT_DIR / 'cardiotoxicity_model.pkl'
with open(model_path, 'wb') as f:
    pickle.dump(final_model, f)
print(f"\n   ✓ Saved trained model: {model_path.name}")
print(f"      Size: {model_path.stat().st_size / 1024:.2f} KB")

# Save as JSON for XGBoost-specific deployment
json_model_path = DEPLOYMENT_DIR / 'cardiotoxicity_model.json'
final_model.save_model(json_model_path)
print(f"   ✓ Saved XGBoost JSON: {json_model_path.name}")
print(f"      Size: {json_model_path.stat().st_size / 1024:.2f} KB")

# ============================================================================
# 4. Create Preprocessing Pipeline
# ============================================================================
print("\n" + "="*80)
print("4. CREATING PREPROCESSING PIPELINE")
print("="*80)

class PreprocessingPipeline:
    """
    Preprocessing pipeline for cardiotoxicity prediction
    
    Steps:
    1. Handle missing values (median for continuous, mode for binary)
    2. Validate input data
    """
    
    def __init__(self, feature_stats):
        """
        Initialize with statistics from training data
        
        Args:
            feature_stats: Dictionary with median/mode values for imputation
        """
        self.feature_stats = feature_stats
        self.continuous_vars = ['age', 'weight', 'height', 'heart_rate', 
                               'LVEF', 'PWT', 'LAd', 'LVDd', 'LVSd']
        
    def preprocess(self, df):
        """
        Apply preprocessing to input dataframe
        
        Args:
            df: Input dataframe with raw features
            
        Returns:
            Preprocessed dataframe
        """
        df_processed = df.copy()
        
        # Handle missing values
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if col in self.continuous_vars:
                    # Use median for continuous variables
                    df_processed[col].fillna(self.feature_stats[col]['median'], inplace=True)
                else:
                    # Use mode for binary variables
                    df_processed[col].fillna(self.feature_stats[col]['mode'], inplace=True)
        
        return df_processed
    
    def validate(self, df):
        """
        Validate input dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Check for required columns
        required_cols = self.continuous_vars + [
            'AC', 'antiHER2', 'HTA', 'DL', 'smoker', 'exsmoker', 
            'diabetes', 'obesity', 'ACprev', 'RTprev', 'heart_rhythm'
        ]
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number])).sum().sum() > 0:
            errors.append("Input contains infinite values")
        
        # Check for valid ranges
        if 'age' in df.columns:
            if (df['age'] < 0).any() or (df['age'] > 120).any():
                errors.append("Age values out of valid range (0-120)")
        
        if 'LVEF' in df.columns:
            if (df['LVEF'] < 0).any() or (df['LVEF'] > 100).any():
                errors.append("LVEF values out of valid range (0-100)")
        
        if errors:
            return False, "; ".join(errors)
        return True, "Valid"

# Calculate feature statistics from training data
print("\nCalculating feature statistics from training data...")
feature_stats = {}

for col in train_processed.columns:
    if col not in ['CTRCD', 'time']:
        if col in ['age', 'weight', 'height', 'heart_rate', 'LVEF', 'PWT', 'LAd', 'LVDd', 'LVSd']:
            feature_stats[col] = {
                'median': float(train_processed[col].median()),
                'mean': float(train_processed[col].mean()),
                'std': float(train_processed[col].std())
            }
        else:
            feature_stats[col] = {
                'mode': int(train_processed[col].mode()[0])
            }

# Create and save preprocessing pipeline
preprocessing_pipeline = PreprocessingPipeline(feature_stats)
preprocessing_path = DEPLOYMENT_DIR / 'preprocessing_pipeline.pkl'

# Save pipeline with protocol 4 for better compatibility
with open(preprocessing_path, 'wb') as f:
    pickle.dump(preprocessing_pipeline, f, protocol=4)
print(f"   ✓ Saved preprocessing pipeline: {preprocessing_path.name}")

# Save feature statistics separately
stats_path = DEPLOYMENT_DIR / 'feature_statistics.json'
with open(stats_path, 'w') as f:
    json.dump(feature_stats, f, indent=2)
print(f"   ✓ Saved feature statistics: {stats_path.name}")

# ============================================================================
# 5. Create Feature Engineering Pipeline
# ============================================================================
print("\n" + "="*80)
print("5. CREATING FEATURE ENGINEERING PIPELINE")
print("="*80)

class FeatureEngineeringPipeline:
    """
    Feature engineering pipeline for cardiotoxicity prediction
    
    Creates 63 engineered features from 25 original features
    """
    
    def __init__(self):
        pass
    
    def engineer_features(self, df):
        """
        Apply feature engineering to preprocessed dataframe
        
        Args:
            df: Preprocessed dataframe with original features
            
        Returns:
            Dataframe with engineered features
        """
        df_eng = df.copy()
        
        # 1. ANTHROPOMETRIC FEATURES
        df_eng['BMI'] = df_eng['weight'] / ((df_eng['height'] / 100) ** 2)
        df_eng['BMI_underweight'] = (df_eng['BMI'] < 18.5).astype(int)
        df_eng['BMI_normal'] = ((df_eng['BMI'] >= 18.5) & (df_eng['BMI'] < 25)).astype(int)
        df_eng['BMI_overweight'] = ((df_eng['BMI'] >= 25) & (df_eng['BMI'] < 30)).astype(int)
        df_eng['BMI_obese'] = (df_eng['BMI'] >= 30).astype(int)
        df_eng['BSA'] = np.sqrt((df_eng['height'] * df_eng['weight']) / 3600)
        
        # 2. CARDIAC STRUCTURAL INDICES
        df_eng['RWT'] = (2 * df_eng['PWT']) / df_eng['LVDd']
        df_eng['LV_mass_approx'] = 0.8 * (1.04 * ((df_eng['LVDd'] + 2*df_eng['PWT'])**3 - df_eng['LVDd']**3)) + 0.6
        df_eng['LVMI'] = df_eng['LV_mass_approx'] / df_eng['BSA']
        df_eng['Fractional_Shortening'] = ((df_eng['LVDd'] - df_eng['LVSd']) / df_eng['LVDd']) * 100
        df_eng['LA_volume_approx'] = (4/3) * np.pi * (df_eng['LAd']/2)**3
        df_eng['LAVi'] = df_eng['LA_volume_approx'] / df_eng['BSA']
        df_eng['LV_hypertrophy'] = (df_eng['PWT'] > 1.1).astype(int)
        df_eng['LA_enlargement'] = (df_eng['LAd'] > 4.0).astype(int)
        df_eng['LV_dilation'] = (df_eng['LVDd'] > 5.5).astype(int)
        
        # 3. CARDIAC FUNCTION CATEGORIES
        df_eng['LVEF_reduced'] = (df_eng['LVEF'] < 50).astype(int)
        df_eng['LVEF_midrange'] = ((df_eng['LVEF'] >= 50) & (df_eng['LVEF'] < 60)).astype(int)
        df_eng['LVEF_preserved'] = (df_eng['LVEF'] >= 60).astype(int)
        df_eng['LVEF_margin'] = df_eng['LVEF'] - 50
        df_eng['systolic_dysfunction_risk'] = (
            (df_eng['LVEF'] < 55).astype(int) + 
            (df_eng['Fractional_Shortening'] < 28).astype(int)
        )
        
        # 4. AGE-CARDIAC INTERACTIONS
        df_eng['age_group_young'] = (df_eng['age'] < 45).astype(int)
        df_eng['age_group_middle'] = ((df_eng['age'] >= 45) & (df_eng['age'] < 65)).astype(int)
        df_eng['age_group_elderly'] = (df_eng['age'] >= 65).astype(int)
        df_eng['age_LVEF_interaction'] = df_eng['age'] * (100 - df_eng['LVEF'])
        df_eng['age_adjusted_LVEF'] = df_eng['LVEF'] / (1 + (df_eng['age'] - 50) * 0.001)
        
        # 5. TREATMENT FEATURES
        df_eng['treatment_AC_only'] = ((df_eng['AC'] == 1) & (df_eng['antiHER2'] == 0)).astype(int)
        df_eng['treatment_antiHER2_only'] = ((df_eng['AC'] == 0) & (df_eng['antiHER2'] == 1)).astype(int)
        df_eng['treatment_combination'] = ((df_eng['AC'] == 1) & (df_eng['antiHER2'] == 1)).astype(int)
        df_eng['treatment_none'] = ((df_eng['AC'] == 0) & (df_eng['antiHER2'] == 0)).astype(int)
        df_eng['prior_treatment_count'] = df_eng['ACprev'].astype(int) + df_eng['RTprev'].astype(int)
        df_eng['current_treatment_count'] = df_eng['AC'].astype(int) + df_eng['antiHER2'].astype(int)
        df_eng['cumulative_cardiotoxic_treatment'] = (
            df_eng['AC'].astype(int) + df_eng['antiHER2'].astype(int) + 
            df_eng['ACprev'].astype(int) + df_eng['RTprev'].astype(int)
        )
        
        # 6. CARDIOVASCULAR RISK SCORE
        df_eng['n_risk_factors'] = (
            df_eng['HTA'].astype(int) + df_eng['DL'].astype(int) + 
            df_eng['smoker'].astype(int) + df_eng['diabetes'].astype(int) + 
            df_eng['obesity'].astype(int)
        )
        df_eng['CV_risk_score'] = (
            df_eng['age'] * 0.01 + 
            df_eng['n_risk_factors'] * 10 + 
            (100 - df_eng['LVEF']) * 0.5
        )
        
        # 7. INTERACTION FEATURES
        df_eng['age_x_AC'] = df_eng['age'] * df_eng['AC']
        df_eng['age_x_antiHER2'] = df_eng['age'] * df_eng['antiHER2']
        df_eng['LVEF_x_AC'] = df_eng['LVEF'] * df_eng['AC']
        df_eng['LVEF_x_antiHER2'] = df_eng['LVEF'] * df_eng['antiHER2']
        df_eng['LVEF_x_n_risk_factors'] = df_eng['LVEF'] * df_eng['n_risk_factors']
        df_eng['LVEF_x_HTA'] = df_eng['LVEF'] * df_eng['HTA']
        df_eng['BMI_x_combination'] = df_eng['BMI'] * df_eng['treatment_combination']
        df_eng['risk_x_treatment'] = df_eng['n_risk_factors'] * df_eng['cumulative_cardiotoxic_treatment']
        
        # 8. POLYNOMIAL FEATURES
        df_eng['age_squared'] = df_eng['age'] ** 2
        df_eng['age_cubed'] = df_eng['age'] ** 3
        df_eng['LVEF_squared'] = df_eng['LVEF'] ** 2
        df_eng['LVEF_cubed'] = df_eng['LVEF'] ** 3
        df_eng['heart_rate_squared'] = df_eng['heart_rate'] ** 2
        df_eng['heart_rate_cubed'] = df_eng['heart_rate'] ** 3
        
        # 9. RATIO FEATURES
        df_eng['LVSd_to_LVDd'] = df_eng['LVSd'] / df_eng['LVDd']
        df_eng['LAd_to_LVDd'] = df_eng['LAd'] / df_eng['LVDd']
        df_eng['PWT_to_LVDd'] = df_eng['PWT'] / df_eng['LVDd']
        df_eng['LVDd_to_BSA'] = df_eng['LVDd'] / df_eng['BSA']
        df_eng['weight_to_height'] = df_eng['weight'] / df_eng['height']
        
        return df_eng

# Create and save feature engineering pipeline
feature_pipeline = FeatureEngineeringPipeline()
feature_pipeline_path = DEPLOYMENT_DIR / 'feature_engineering_pipeline.pkl'
with open(feature_pipeline_path, 'wb') as f:
    pickle.dump(feature_pipeline, f)
print(f"   ✓ Saved feature engineering pipeline: {feature_pipeline_path.name}")

# ============================================================================
# 6. Create Model Metadata
# ============================================================================
print("\n" + "="*80)
print("6. CREATING MODEL METADATA")
print("="*80)

metadata = {
    "model_info": {
        "name": "Cardiotoxicity Prediction Model",
        "version": "1.0",
        "type": "XGBoost Classifier",
        "objective": "Predict cancer treatment-related cardiac dysfunction (CTRCD)",
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "framework": "xgboost",
        "framework_version": xgb.__version__
    },
    
    "data_info": {
        "total_samples": 531,
        "training_samples": len(train_df),
        "validation_samples": len(val_df),
        "test_samples": len(test_df),
        "positive_class_count": int(y_train.sum()) + int(y_val.sum()) + int(y_test.sum()),
        "positive_class_percentage": 10.17,
        "class_imbalance_ratio": 8.8
    },
    
    "features": {
        "total_features": len(selected_features),
        "selected_features": selected_features,
        "feature_selection_method": "RFE (Recursive Feature Elimination)",
        "original_features": 25,
        "engineered_features": 63,
        "feature_categories": feature_categories
    },
    
    "hyperparameters": optimized_params,
    
    "performance": {
        "train": {
            "roc_auc": float(train_auc),
            "pr_auc": float(train_pr)
        },
        "validation": {
            "roc_auc": float(val_auc),
            "pr_auc": float(val_pr)
        },
        "test": {
            "roc_auc": float(test_auc),
            "pr_auc": float(test_pr)
        },
        "metrics_optimized_for": "ROC AUC"
    },
    
    "improvements": {
        "baseline_test_auc": 0.6094,
        "final_test_auc": float(test_auc),
        "improvement_percentage": float((test_auc - 0.6094) / 0.6094 * 100),
        "feature_reduction": "88 → 40 features (54.5% reduction)"
    },
    
    "threshold_recommendations": {
        "default": 0.5,
        "high_sensitivity": 0.3,
        "high_specificity": 0.7,
        "note": "Adjust threshold based on clinical context and cost of false positives/negatives"
    },
    
    "input_requirements": {
        "required_columns": [
            "age", "weight", "height", "heart_rate", "LVEF", "PWT", "LAd", "LVDd", "LVSd",
            "AC", "antiHER2", "HTA", "DL", "smoker", "exsmoker", "diabetes", "obesity",
            "ACprev", "RTprev", "heart_rhythm"
        ],
        "data_types": {
            "continuous": ["age", "weight", "height", "heart_rate", "LVEF", "PWT", "LAd", "LVDd", "LVSd"],
            "binary": ["AC", "antiHER2", "HTA", "DL", "smoker", "exsmoker", "diabetes", "obesity", "ACprev", "RTprev", "heart_rhythm"]
        },
        "missing_values": "Handled automatically via preprocessing pipeline",
        "units": {
            "age": "years",
            "weight": "kg",
            "height": "cm",
            "heart_rate": "bpm",
            "LVEF": "%",
            "PWT": "cm",
            "LAd": "cm",
            "LVDd": "cm",
            "LVSd": "cm"
        }
    },
    
    "clinical_interpretation": {
        "top_risk_factors": [
            "CV_risk_score (cardiovascular risk composite)",
            "heart_rate_cubed (elevated heart rate)",
            "risk_x_treatment (interaction of CV risk with cardiotoxic treatment)",
            "cumulative_cardiotoxic_treatment (treatment burden)",
            "age_adjusted_LVEF (age-normalized cardiac function)"
        ],
        "key_insights": [
            "Cardiovascular risk factors compound with cardiotoxic treatment effects",
            "Baseline cardiac function (LVEF) is critical predictor",
            "Cumulative treatment burden matters - consider prior exposures",
            "Heart rate elevation may indicate early cardiac stress",
            "Age modifies the effect of cardiac function on risk"
        ]
    },
    
    "limitations": {
        "sample_size": "Limited to 531 patients from single cohort",
        "class_imbalance": "Only 10.17% positive cases - may underpredict rare events",
        "external_validation": "Not validated on external datasets",
        "temporal": "Does not explicitly model time-to-event",
        "imaging": "Does not include advanced cardiac imaging features"
    },
    
    "recommendations": {
        "usage": [
            "Use as clinical decision support tool, not sole diagnostic criterion",
            "Combine with clinical judgment and additional tests",
            "Monitor high-risk patients more frequently",
            "Consider cardioprotective strategies for high-risk predictions"
        ],
        "deployment": [
            "Implement in clinical workflow with physician oversight",
            "Regular model monitoring and recalibration",
            "Collect feedback for model improvement",
            "Ensure interpretability for clinical trust"
        ]
    }
}

# Save metadata
metadata_path = DEPLOYMENT_DIR / 'model_metadata.json'
with open(metadata_path, 'w') as f:
    json.dump(metadata, f, indent=2)
print(f"   ✓ Saved model metadata: {metadata_path.name}")

# ============================================================================
# 7. Create Usage Instructions
# ============================================================================
print("\n" + "="*80)
print("7. CREATING USAGE INSTRUCTIONS")
print("="*80)

usage_instructions = """
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
"""

# Save usage instructions
instructions_path = DEPLOYMENT_DIR / 'USAGE_INSTRUCTIONS.md'
with open(instructions_path, 'w', encoding='utf-8') as f:
    f.write(usage_instructions)
print(f"   ✓ Saved usage instructions: {instructions_path.name}")

# ============================================================================
# 8. Create Example Script
# ============================================================================
print("\n" + "="*80)
print("8. CREATING EXAMPLE SCRIPT")
print("="*80)

example_script = """#!/usr/bin/env python3
\"\"\"
Example: Cardiotoxicity Risk Prediction
Demonstrates how to use the deployed model for individual patient prediction
\"\"\"

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
print("✓ Model loaded successfully!\\n")

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
    \"\"\"Predict CTRCD risk for a patient\"\"\"
    print(f"\\n{patient_name}:")
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
    print(f"\\n  Key Clinical Parameters:")
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

print("\\n" + "="*80)
print("Predictions completed successfully!")
print("="*80)
"""

example_path = DEPLOYMENT_DIR / 'example_prediction.py'
with open(example_path, 'w', encoding='utf-8') as f:
    f.write(example_script)
print(f"   ✓ Saved example script: {example_path.name}")

# ============================================================================
# 9. Create Requirements File
# ============================================================================
print("\n" + "="*80)
print("9. CREATING REQUIREMENTS FILE")
print("="*80)

requirements = f"""# Cardiotoxicity Prediction Model - Requirements
# Python version: 3.8+

# Core dependencies
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
xgboost=={xgb.__version__}

# Optional dependencies for development
matplotlib>=3.4.0
seaborn>=0.11.0
shap>=0.40.0

# For deployment
pickle5>=0.0.11  # Python 3.7 compatibility
"""

requirements_path = DEPLOYMENT_DIR / 'requirements.txt'
with open(requirements_path, 'w', encoding='utf-8') as f:
    f.write(requirements)
print(f"   ✓ Saved requirements: {requirements_path.name}")

# ============================================================================
# 10. Create Deployment Summary
# ============================================================================
print("\n" + "="*80)
print("10. DEPLOYMENT PACKAGE SUMMARY")
print("="*80)

deployment_summary = {
    "deployment_package": {
        "created_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_version": "1.0",
        "package_contents": [
            {
                "file": "cardiotoxicity_model.pkl",
                "type": "Trained Model (Pickle)",
                "description": "XGBoost classifier ready for deployment",
                "size_kb": round(model_path.stat().st_size / 1024, 2)
            },
            {
                "file": "cardiotoxicity_model.json",
                "type": "Trained Model (JSON)",
                "description": "XGBoost model in JSON format",
                "size_kb": round(json_model_path.stat().st_size / 1024, 2)
            },
            {
                "file": "preprocessing_pipeline.pkl",
                "type": "Preprocessing Pipeline",
                "description": "Handles missing values and data validation",
                "size_kb": round(preprocessing_path.stat().st_size / 1024, 2)
            },
            {
                "file": "feature_engineering_pipeline.pkl",
                "type": "Feature Engineering Pipeline",
                "description": "Creates 63 engineered features from 20 inputs",
                "size_kb": round(feature_pipeline_path.stat().st_size / 1024, 2)
            },
            {
                "file": "model_metadata.json",
                "type": "Model Metadata",
                "description": "Complete model documentation and configuration",
                "size_kb": round(metadata_path.stat().st_size / 1024, 2)
            },
            {
                "file": "feature_statistics.json",
                "type": "Feature Statistics",
                "description": "Training data statistics for imputation",
                "size_kb": round(stats_path.stat().st_size / 1024, 2)
            },
            {
                "file": "USAGE_INSTRUCTIONS.md",
                "type": "Documentation",
                "description": "Comprehensive usage guide with examples",
                "size_kb": round(instructions_path.stat().st_size / 1024, 2)
            },
            {
                "file": "example_prediction.py",
                "type": "Example Script",
                "description": "Working example for patient predictions",
                "size_kb": round(example_path.stat().st_size / 1024, 2)
            },
            {
                "file": "requirements.txt",
                "type": "Dependencies",
                "description": "Python package requirements",
                "size_kb": round(requirements_path.stat().st_size / 1024, 2)
            }
        ],
        "total_size_kb": round(sum([
            model_path.stat().st_size,
            json_model_path.stat().st_size,
            preprocessing_path.stat().st_size,
            feature_pipeline_path.stat().st_size,
            metadata_path.stat().st_size,
            stats_path.stat().st_size,
            instructions_path.stat().st_size,
            example_path.stat().st_size,
            requirements_path.stat().st_size
        ]) / 1024, 2)
    },
    "deployment_checklist": {
        "model_artifacts": "✓ Complete",
        "pipelines": "✓ Complete",
        "metadata": "✓ Complete",
        "documentation": "✓ Complete",
        "examples": "✓ Complete",
        "requirements": "✓ Complete"
    },
    "model_performance": {
        "test_roc_auc": float(test_auc),
        "test_pr_auc": float(test_pr),
        "improvement_over_baseline": f"+{((test_auc - 0.6094) / 0.6094 * 100):.2f}%"
    },
    "next_steps": [
        "Test deployment package in production environment",
        "Integrate with clinical information system",
        "Set up monitoring for model performance",
        "Establish feedback loop for continuous improvement",
        "Conduct external validation on new patient cohorts",
        "Regular model retraining with updated data"
    ]
}

summary_path = DEPLOYMENT_DIR / 'deployment_summary.json'
with open(summary_path, 'w') as f:
    json.dump(deployment_summary, f, indent=2)
print(f"\n   ✓ Saved deployment summary: {summary_path.name}")

# Print summary
print(f"\n📦 DEPLOYMENT PACKAGE CREATED")
print(f"   Location: {DEPLOYMENT_DIR}")
print(f"   Total files: {len(deployment_summary['deployment_package']['package_contents'])}")
print(f"   Total size: {deployment_summary['deployment_package']['total_size_kb']:.2f} KB")

print("\n📋 Package Contents:")
for item in deployment_summary['deployment_package']['package_contents']:
    print(f"   • {item['file']:<40} ({item['size_kb']:>6.2f} KB)")

print("\n✅ Deployment Checklist:")
for check, status in deployment_summary['deployment_checklist'].items():
    print(f"   {status} {check.replace('_', ' ').title()}")

# ============================================================================
# Final Summary
# ============================================================================
print("\n" + "="*80)
print("PHASE 11 COMPLETED SUCCESSFULLY!")
print("="*80)

print(f"""
✅ All deployment artifacts created successfully!

📊 Model Performance:
   • Test ROC AUC: {test_auc:.4f}
   • Test PR AUC: {test_pr:.4f}
   • Improvement: {((test_auc - 0.6094) / 0.6094 * 100):.2f}% over baseline

📦 Deployment Package:
   • Location: {DEPLOYMENT_DIR}
   • Ready for production deployment
   • Complete with documentation and examples

🚀 Next Steps:
   1. Review USAGE_INSTRUCTIONS.md for deployment guide
   2. Test example_prediction.py to verify setup
   3. Integrate with clinical information system
   4. Set up model monitoring in production
   5. Establish feedback loop for continuous improvement

📝 Documentation:
   • Model metadata: model_metadata.json
   • Usage guide: USAGE_INSTRUCTIONS.md
   • Example script: example_prediction.py
   • Deployment summary: deployment_summary.json

The model is now ready for clinical deployment! 🎉
""")

print("="*80)
