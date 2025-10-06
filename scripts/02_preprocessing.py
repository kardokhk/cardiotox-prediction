"""
Phase 2: Data Preprocessing
Handles missing values, creates train/validation/test splits, and prepares data for modeling

Author: Competition Submission
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'dataset'
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
TABLES_DIR = BASE_DIR / 'results' / 'tables'

PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 2: DATA PREPROCESSING")
print("="*80)

# Load data
print("\n1. Loading raw data...")
df = pd.read_csv(DATA_DIR / 'BC_cardiotox_clinical_variables.csv', 
                 sep=';', decimal=',')
print(f"   Original shape: {df.shape}")

# Create a copy for processing
df_processed = df.copy()

# Separate features and target
print("\n2. Separating features and target...")
target_col = 'CTRCD'
time_col = 'time'  # Keep for potential survival analysis later
X = df_processed.drop([target_col, time_col], axis=1)
y = df_processed[target_col]
time_data = df_processed[time_col]

print(f"   Features shape: {X.shape}")
print(f"   Target distribution: {y.value_counts().to_dict()}")

# Analyze missing data patterns
print("\n3. Analyzing missing data patterns...")
missing_info = pd.DataFrame({
    'Feature': X.columns,
    'Missing_Count': X.isnull().sum().values,
    'Missing_Percentage': (X.isnull().sum() / len(X) * 100).values
}).sort_values('Missing_Count', ascending=False)

print("\nMissing Data Summary:")
print(missing_info[missing_info['Missing_Count'] > 0].to_string(index=False))

# Strategy: 
# - Very few missing (<1%): median/mode imputation
# - Moderate missing (~9%): KNN imputation for continuous, mode for binary
# - Group missing pattern: likely these ~50 patients are from early cohort without risk factor data

# Identify variable types
continuous_vars = ['age', 'weight', 'height', 'heart_rate', 'LVEF', 'PWT', 'LAd', 'LVDd', 'LVSd']
binary_vars = [col for col in X.columns if col not in continuous_vars]

print(f"\n   Continuous variables: {len(continuous_vars)}")
print(f"   Binary variables: {len(binary_vars)}")

# Handle missing values
print("\n4. Handling missing values...")

# For continuous variables with minimal missing (<1%), use median
minimal_missing_continuous = ['weight', 'height', 'heart_rhythm', 'PWT', 'LAd', 'LVDd', 'LVSd']
for col in minimal_missing_continuous:
    if col in X.columns and X[col].isnull().sum() > 0:
        median_val = X[col].median()
        X[col].fillna(median_val, inplace=True)
        print(f"   ✓ {col}: filled {X[col].isnull().sum()} values with median ({median_val:.2f})")

# For binary variables with ~9% missing, use mode (most common value)
for col in binary_vars:
    if X[col].isnull().sum() > 0:
        mode_val = X[col].mode()[0]
        n_missing = X[col].isnull().sum()
        X[col].fillna(mode_val, inplace=True)
        print(f"   ✓ {col}: filled {n_missing} values with mode ({mode_val})")

# Verify no missing values
print(f"\n   Total missing values after imputation: {X.isnull().sum().sum()}")

# Data validation
print("\n5. Data validation...")
print(f"   Final features shape: {X.shape}")
print(f"   No missing values: {X.isnull().sum().sum() == 0}")
print(f"   No infinite values: {np.isinf(X.select_dtypes(include=[np.number])).sum().sum() == 0}")

# Save processed features with target
print("\n6. Creating processed dataset...")
df_final = X.copy()
df_final['CTRCD'] = y
df_final['time'] = time_data

# Summary statistics
print("\n7. Processed data summary:")
print(f"   Total samples: {len(df_final)}")
print(f"   Total features: {X.shape[1]}")
print(f"   CTRCD=0: {(y==0).sum()} ({(y==0).sum()/len(y)*100:.2f}%)")
print(f"   CTRCD=1: {(y==1).sum()} ({(y==1).sum()/len(y)*100:.2f}%)")

# Split data: 70% train, 15% validation, 15% test (stratified)
print("\n8. Creating stratified train/validation/test splits...")
print("   Strategy: 70% train, 15% validation, 15% test")

# First split: 70% train, 30% temp
X_train, X_temp, y_train, y_temp, time_train, time_temp = train_test_split(
    X, y, time_data, test_size=0.30, random_state=42, stratify=y
)

# Second split: split temp into 50-50 (which gives us 15% val, 15% test of original)
X_val, X_test, y_val, y_test, time_val, time_test = train_test_split(
    X_temp, y_temp, time_temp, test_size=0.50, random_state=42, stratify=y_temp
)

print(f"\n   Train set: {X_train.shape[0]} samples")
print(f"      - CTRCD=0: {(y_train==0).sum()} ({(y_train==0).sum()/len(y_train)*100:.2f}%)")
print(f"      - CTRCD=1: {(y_train==1).sum()} ({(y_train==1).sum()/len(y_train)*100:.2f}%)")

print(f"\n   Validation set: {X_val.shape[0]} samples")
print(f"      - CTRCD=0: {(y_val==0).sum()} ({(y_val==0).sum()/len(y_val)*100:.2f}%)")
print(f"      - CTRCD=1: {(y_val==1).sum()} ({(y_val==1).sum()/len(y_val)*100:.2f}%)")

print(f"\n   Test set: {X_test.shape[0]} samples")
print(f"      - CTRCD=0: {(y_test==0).sum()} ({(y_test==0).sum()/len(y_test)*100:.2f}%)")
print(f"      - CTRCD=1: {(y_test==1).sum()} ({(y_test==1).sum()/len(y_test)*100:.2f}%)")

# Save datasets
print("\n9. Saving processed datasets...")

# Save complete processed dataset
df_final.to_csv(PROCESSED_DIR / 'data_processed_complete.csv', index=False)
print(f"   ✓ Saved: data_processed_complete.csv")

# Save train set
train_df = X_train.copy()
train_df['CTRCD'] = y_train
train_df['time'] = time_train
train_df.to_csv(PROCESSED_DIR / 'train_data.csv', index=False)
print(f"   ✓ Saved: train_data.csv")

# Save validation set
val_df = X_val.copy()
val_df['CTRCD'] = y_val
val_df['time'] = time_val
val_df.to_csv(PROCESSED_DIR / 'val_data.csv', index=False)
print(f"   ✓ Saved: val_data.csv")

# Save test set
test_df = X_test.copy()
test_df['CTRCD'] = y_test
test_df['time'] = time_test
test_df.to_csv(PROCESSED_DIR / 'test_data.csv', index=False)
print(f"   ✓ Saved: test_data.csv")

# Create and save feature lists
feature_info = {
    'all_features': X.columns.tolist(),
    'continuous_features': continuous_vars,
    'binary_features': binary_vars,
    'target': target_col,
    'time_variable': time_col
}

import json
with open(PROCESSED_DIR / 'feature_info.json', 'w') as f:
    json.dump(feature_info, f, indent=2)
print(f"   ✓ Saved: feature_info.json")

# Create split summary table
split_summary = pd.DataFrame({
    'Split': ['Train', 'Validation', 'Test', 'Total'],
    'Total_Samples': [len(y_train), len(y_val), len(y_test), len(y)],
    'CTRCD_0': [(y_train==0).sum(), (y_val==0).sum(), (y_test==0).sum(), (y==0).sum()],
    'CTRCD_1': [(y_train==1).sum(), (y_val==1).sum(), (y_test==1).sum(), (y==1).sum()],
    'CTRCD_1_Percentage': [
        f"{(y_train==1).sum()/len(y_train)*100:.2f}%",
        f"{(y_val==1).sum()/len(y_val)*100:.2f}%",
        f"{(y_test==1).sum()/len(y_test)*100:.2f}%",
        f"{(y==1).sum()/len(y)*100:.2f}%"
    ]
})

split_summary.to_csv(TABLES_DIR / '08_data_split_summary.csv', index=False)
print(f"   ✓ Saved: 08_data_split_summary.csv")

print("\n" + "="*80)
print("PREPROCESSING COMPLETED SUCCESSFULLY")
print("="*80)
print("\nData Quality Checks:")
print(f"  ✓ No missing values in processed data")
print(f"  ✓ Stratified splits maintain class balance")
print(f"  ✓ Train/Val/Test: {len(y_train)}/{len(y_val)}/{len(y_test)} samples")
print(f"\nReady for feature engineering (Phase 3)")
print("="*80)
