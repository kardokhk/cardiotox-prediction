"""
Phase 3: Feature Engineering
Creates domain-specific features for cardiotoxicity prediction

Key concepts from cardiology literature:
- BMI and body composition
- Cardiac structural indices (relative wall thickness, etc.)
- Treatment combinations and cumulative effects
- Age-cardiac function interactions
- Cardiovascular risk scores

Author: Competition Submission
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
PROCESSED_DIR = BASE_DIR / 'data' / 'processed'
ENGINEERED_DIR = BASE_DIR / 'data' / 'engineered'
TABLES_DIR = BASE_DIR / 'results' / 'tables'

ENGINEERED_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 3: FEATURE ENGINEERING")
print("="*80)

def create_engineered_features(df, feature_set='train'):
    """
    Create domain-specific engineered features
    
    Args:
        df: Input dataframe
        feature_set: 'train', 'val', or 'test' for tracking
    
    Returns:
        DataFrame with engineered features
    """
    df_eng = df.copy()
    
    print(f"\n{feature_set.upper()} SET - Creating engineered features...")
    
    # ============================================================================
    # 1. ANTHROPOMETRIC FEATURES
    # ============================================================================
    print("  1. Anthropometric features...")
    
    # Body Mass Index (BMI) - key cardiotoxicity risk factor
    df_eng['BMI'] = df_eng['weight'] / ((df_eng['height'] / 100) ** 2)
    
    # BMI categories (WHO classification)
    df_eng['BMI_underweight'] = (df_eng['BMI'] < 18.5).astype(int)
    df_eng['BMI_normal'] = ((df_eng['BMI'] >= 18.5) & (df_eng['BMI'] < 25)).astype(int)
    df_eng['BMI_overweight'] = ((df_eng['BMI'] >= 25) & (df_eng['BMI'] < 30)).astype(int)
    df_eng['BMI_obese'] = (df_eng['BMI'] >= 30).astype(int)
    
    # Body Surface Area (BSA) - Mosteller formula
    df_eng['BSA'] = np.sqrt((df_eng['height'] * df_eng['weight']) / 3600)
    
    print(f"     ✓ Created BMI (mean: {df_eng['BMI'].mean():.2f})")
    print(f"     ✓ Created BMI categories")
    print(f"     ✓ Created BSA")
    
    # ============================================================================
    # 2. CARDIAC STRUCTURAL INDICES
    # ============================================================================
    print("  2. Cardiac structural indices...")
    
    # Relative Wall Thickness (RWT) - indicator of cardiac geometry
    # RWT = (2 × PWT) / LVDd
    df_eng['RWT'] = (2 * df_eng['PWT']) / df_eng['LVDd']
    
    # Left Ventricular Mass Index (simplified approximation)
    # LVMI ≈ 0.8 × (1.04 × [(LVDd + PWT + IVS)³ - LVDd³]) + 0.6
    # Simplified: using PWT as proxy for both walls
    df_eng['LV_mass_approx'] = 0.8 * (1.04 * ((df_eng['LVDd'] + 2*df_eng['PWT'])**3 - df_eng['LVDd']**3)) + 0.6
    df_eng['LVMI'] = df_eng['LV_mass_approx'] / df_eng['BSA']
    
    # Fractional Shortening (FS) = (LVDd - LVSd) / LVDd × 100
    df_eng['Fractional_Shortening'] = ((df_eng['LVDd'] - df_eng['LVSd']) / df_eng['LVDd']) * 100
    
    # Left Atrial Volume Index (simplified)
    # LAVi approximation using LA diameter
    df_eng['LA_volume_approx'] = (4/3) * np.pi * (df_eng['LAd']/2)**3
    df_eng['LAVi'] = df_eng['LA_volume_approx'] / df_eng['BSA']
    
    # Cardiac structural abnormality indicators
    df_eng['LV_hypertrophy'] = (df_eng['PWT'] > 1.1).astype(int)  # Increased wall thickness
    df_eng['LA_enlargement'] = (df_eng['LAd'] > 4.0).astype(int)  # LA diameter > 4cm
    df_eng['LV_dilation'] = (df_eng['LVDd'] > 5.5).astype(int)    # LV dilation
    
    print(f"     ✓ Created RWT (mean: {df_eng['RWT'].mean():.3f})")
    print(f"     ✓ Created LVMI, Fractional Shortening, LAVi")
    print(f"     ✓ Created structural abnormality flags")
    
    # ============================================================================
    # 3. CARDIAC FUNCTION CATEGORIES
    # ============================================================================
    print("  3. Cardiac function categories...")
    
    # LVEF categories (ESC classification)
    df_eng['LVEF_reduced'] = (df_eng['LVEF'] < 50).astype(int)        # HFrEF
    df_eng['LVEF_midrange'] = ((df_eng['LVEF'] >= 50) & (df_eng['LVEF'] < 60)).astype(int)  # HFmrEF
    df_eng['LVEF_preserved'] = (df_eng['LVEF'] >= 60).astype(int)     # HFpEF
    
    # Distance from CTRCD threshold (LVEF = 50%)
    df_eng['LVEF_margin'] = df_eng['LVEF'] - 50
    
    # Systolic function composite
    df_eng['systolic_dysfunction_risk'] = (
        (df_eng['LVEF'] < 55).astype(int) + 
        (df_eng['Fractional_Shortening'] < 28).astype(int)
    )
    
    print(f"     ✓ Created LVEF categories")
    print(f"     ✓ Created LVEF margin and dysfunction risk")
    
    # ============================================================================
    # 4. AGE-CARDIAC INTERACTIONS
    # ============================================================================
    print("  4. Age-cardiac function interactions...")
    
    # Age groups
    df_eng['age_group_young'] = (df_eng['age'] < 45).astype(int)
    df_eng['age_group_middle'] = ((df_eng['age'] >= 45) & (df_eng['age'] < 65)).astype(int)
    df_eng['age_group_elderly'] = (df_eng['age'] >= 65).astype(int)
    
    # Age-LVEF interaction (older patients with lower LVEF at higher risk)
    df_eng['age_LVEF_interaction'] = df_eng['age'] * (100 - df_eng['LVEF'])
    
    # Age-adjusted LVEF (normalized)
    df_eng['age_adjusted_LVEF'] = df_eng['LVEF'] / (1 + (df_eng['age'] - 50) * 0.001)
    
    print(f"     ✓ Created age groups and age-LVEF interactions")
    
    # ============================================================================
    # 5. TREATMENT FEATURES
    # ============================================================================
    print("  5. Treatment combinations...")
    
    # Current treatment combinations
    df_eng['treatment_AC_only'] = ((df_eng['AC'] == 1) & (df_eng['antiHER2'] == 0)).astype(int)
    df_eng['treatment_antiHER2_only'] = ((df_eng['AC'] == 0) & (df_eng['antiHER2'] == 1)).astype(int)
    df_eng['treatment_combination'] = ((df_eng['AC'] == 1) & (df_eng['antiHER2'] == 1)).astype(int)
    df_eng['treatment_none'] = ((df_eng['AC'] == 0) & (df_eng['antiHER2'] == 0)).astype(int)
    
    # Cumulative treatment exposure
    df_eng['cumulative_cardiotoxic_treatment'] = (
        df_eng['AC'] + df_eng['antiHER2'] + 
        df_eng['ACprev'] + df_eng['antiHER2prev']
    )
    
    # Re-treatment indicator
    df_eng['retreatment_AC'] = ((df_eng['AC'] == 1) & (df_eng['ACprev'] == 1)).astype(int)
    df_eng['retreatment_antiHER2'] = ((df_eng['antiHER2'] == 1) & (df_eng['antiHER2prev'] == 1)).astype(int)
    df_eng['any_retreatment'] = (df_eng['retreatment_AC'] + df_eng['retreatment_antiHER2'] > 0).astype(int)
    
    # Previous treatment with current exposure
    df_eng['prior_treatment_count'] = df_eng['ACprev'] + df_eng['antiHER2prev']
    
    print(f"     ✓ Created treatment combinations")
    print(f"     ✓ Created cumulative exposure features")
    
    # ============================================================================
    # 6. CARDIOVASCULAR RISK SCORE
    # ============================================================================
    print("  6. Cardiovascular risk scoring...")
    
    # HFA-ICOS inspired risk score (simplified)
    # Based on: age, hypertension, diabetes, baseline LVEF, previous CV disease
    risk_score = 0
    
    # Age component
    risk_score += (df_eng['age'] >= 65).astype(int) * 2
    risk_score += ((df_eng['age'] >= 50) & (df_eng['age'] < 65)).astype(int) * 1
    
    # Cardiac function component
    risk_score += (df_eng['LVEF'] < 50).astype(int) * 3
    risk_score += ((df_eng['LVEF'] >= 50) & (df_eng['LVEF'] < 60)).astype(int) * 2
    
    # Risk factors component
    risk_score += df_eng['HTA'].astype(int) * 1
    risk_score += df_eng['DM'].astype(int) * 2
    risk_score += df_eng['DL'].astype(int) * 1
    
    # Previous cardiac history component
    risk_score += df_eng['CIprev'].astype(int) * 3
    risk_score += df_eng['ICMprev'].astype(int) * 2
    risk_score += df_eng['ARRprev'].astype(int) * 1
    risk_score += df_eng['VALVprev'].astype(int) * 1
    
    # Previous cancer treatment component
    risk_score += df_eng['ACprev'].astype(int) * 2
    risk_score += df_eng['antiHER2prev'].astype(int) * 2
    risk_score += df_eng['RTprev'].astype(int) * 1
    
    df_eng['CV_risk_score'] = risk_score
    
    # Risk categories
    df_eng['CV_risk_low'] = (df_eng['CV_risk_score'] <= 3).astype(int)
    df_eng['CV_risk_moderate'] = ((df_eng['CV_risk_score'] > 3) & (df_eng['CV_risk_score'] <= 7)).astype(int)
    df_eng['CV_risk_high'] = ((df_eng['CV_risk_score'] > 7) & (df_eng['CV_risk_score'] <= 12)).astype(int)
    df_eng['CV_risk_very_high'] = (df_eng['CV_risk_score'] > 12).astype(int)
    
    print(f"     ✓ Created CV risk score (mean: {df_eng['CV_risk_score'].mean():.2f})")
    print(f"     ✓ Created risk categories")
    
    # ============================================================================
    # 7. COMORBIDITY FEATURES
    # ============================================================================
    print("  7. Comorbidity features...")
    
    # Total number of cardiovascular risk factors
    df_eng['n_CV_risk_factors'] = (
        df_eng['HTA'] + df_eng['DL'] + df_eng['DM'] + 
        df_eng['smoker'] + df_eng['exsmoker']
    )
    
    # Metabolic syndrome indicators (simplified)
    df_eng['metabolic_syndrome_risk'] = (
        ((df_eng['BMI'] >= 30).astype(int)) + 
        df_eng['HTA'] + 
        df_eng['DL'] + 
        df_eng['DM']
    )
    
    # Previous cardiac history count
    df_eng['n_previous_cardiac_events'] = (
        df_eng['CIprev'] + df_eng['ICMprev'] + 
        df_eng['ARRprev'] + df_eng['VALVprev'] + df_eng['cxvalv']
    )
    
    # Any previous cardiac history flag
    df_eng['any_cardiac_history'] = (df_eng['n_previous_cardiac_events'] > 0).astype(int)
    
    print(f"     ✓ Created comorbidity counts")
    
    # ============================================================================
    # 8. INTERACTION FEATURES
    # ============================================================================
    print("  8. High-order interaction features...")
    
    # Age × Treatment
    df_eng['age_x_AC'] = df_eng['age'] * df_eng['AC']
    df_eng['age_x_antiHER2'] = df_eng['age'] * df_eng['antiHER2']
    
    # LVEF × Treatment (lower LVEF + treatment = higher risk)
    df_eng['LVEF_x_AC'] = df_eng['LVEF'] * df_eng['AC']
    df_eng['LVEF_x_antiHER2'] = df_eng['LVEF'] * df_eng['antiHER2']
    
    # BMI × Treatment
    df_eng['BMI_x_combination'] = df_eng['BMI'] * df_eng['treatment_combination']
    
    # Risk score × Treatment
    df_eng['risk_x_treatment'] = df_eng['CV_risk_score'] * df_eng['cumulative_cardiotoxic_treatment']
    
    # Age × Risk factors
    df_eng['age_x_HTA'] = df_eng['age'] * df_eng['HTA']
    df_eng['age_x_DM'] = df_eng['age'] * df_eng['DM']
    
    # LVEF × Risk factors
    df_eng['LVEF_x_HTA'] = df_eng['LVEF'] * df_eng['HTA']
    df_eng['LVEF_x_n_risk_factors'] = df_eng['LVEF'] * df_eng['n_CV_risk_factors']
    
    print(f"     ✓ Created interaction features")
    
    # ============================================================================
    # 9. POLYNOMIAL FEATURES FOR KEY VARIABLES
    # ============================================================================
    print("  9. Polynomial features...")
    
    # Key variables: age, LVEF, heart_rate (squared and cubed for non-linear relationships)
    for var in ['age', 'LVEF', 'heart_rate']:
        df_eng[f'{var}_squared'] = df_eng[var] ** 2
        df_eng[f'{var}_cubed'] = df_eng[var] ** 3
    
    print(f"     ✓ Created polynomial features for age, LVEF, heart_rate")
    
    # ============================================================================
    # 10. RATIO FEATURES
    # ============================================================================
    print("  10. Ratio features...")
    
    # Cardiac geometry ratios
    df_eng['LVDd_to_BSA'] = df_eng['LVDd'] / df_eng['BSA']
    df_eng['LAd_to_LVDd'] = df_eng['LAd'] / df_eng['LVDd']
    df_eng['LVSd_to_LVDd'] = df_eng['LVSd'] / df_eng['LVDd']
    
    # Weight to height ratio
    df_eng['weight_to_height'] = df_eng['weight'] / df_eng['height']
    
    print(f"     ✓ Created ratio features")
    
    # ============================================================================
    # FEATURE SUMMARY
    # ============================================================================
    n_original = len([c for c in df.columns if c not in ['CTRCD', 'time']])
    n_engineered = len([c for c in df_eng.columns if c not in df.columns and c not in ['CTRCD', 'time']])
    n_total = len([c for c in df_eng.columns if c not in ['CTRCD', 'time']])
    
    print(f"\n  Feature Summary:")
    print(f"     Original features: {n_original}")
    print(f"     Engineered features: {n_engineered}")
    print(f"     Total features: {n_total}")
    
    return df_eng

# ============================================================================
# MAIN PROCESSING
# ============================================================================

print("\n1. Loading processed data...")
train_df = pd.read_csv(PROCESSED_DIR / 'train_data.csv')
val_df = pd.read_csv(PROCESSED_DIR / 'val_data.csv')
test_df = pd.read_csv(PROCESSED_DIR / 'test_data.csv')

print(f"   Train: {train_df.shape}")
print(f"   Val: {val_df.shape}")
print(f"   Test: {test_df.shape}")

# Create engineered features for each set
print("\n2. Feature engineering...")
train_eng = create_engineered_features(train_df, 'train')
val_eng = create_engineered_features(val_df, 'val')
test_eng = create_engineered_features(test_df, 'test')

# Save engineered datasets
print("\n3. Saving engineered datasets...")
train_eng.to_csv(ENGINEERED_DIR / 'train_engineered.csv', index=False)
val_eng.to_csv(ENGINEERED_DIR / 'val_engineered.csv', index=False)
test_eng.to_csv(ENGINEERED_DIR / 'test_engineered.csv', index=False)

print(f"   ✓ Saved: train_engineered.csv ({train_eng.shape})")
print(f"   ✓ Saved: val_engineered.csv ({val_eng.shape})")
print(f"   ✓ Saved: test_engineered.csv ({test_eng.shape})")

# Create feature list
original_features = [c for c in train_df.columns if c not in ['CTRCD', 'time']]
engineered_features = [c for c in train_eng.columns if c not in train_df.columns and c not in ['CTRCD', 'time']]
all_features = [c for c in train_eng.columns if c not in ['CTRCD', 'time']]

feature_categories = {
    'original': original_features,
    'engineered': engineered_features,
    'all': all_features,
    'anthropometric': [c for c in engineered_features if 'BMI' in c or 'BSA' in c or 'weight' in c],
    'cardiac_structure': [c for c in engineered_features if any(x in c for x in ['RWT', 'LVMI', 'Fractional', 'LAVi', 'LV_', 'LA_'])],
    'cardiac_function': [c for c in engineered_features if 'LVEF' in c and c != 'LVEF'],
    'age_related': [c for c in engineered_features if 'age' in c.lower()],
    'treatment': [c for c in engineered_features if 'treatment' in c or 'cumulative' in c or 'retreatment' in c],
    'risk_score': [c for c in engineered_features if 'risk' in c.lower() or 'CV' in c],
    'comorbidity': [c for c in engineered_features if 'n_' in c or 'metabolic' in c or 'cardiac_history' in c],
    'interactions': [c for c in engineered_features if '_x_' in c],
    'polynomial': [c for c in engineered_features if 'squared' in c or 'cubed' in c],
    'ratios': [c for c in engineered_features if '_to_' in c]
}

import json
with open(ENGINEERED_DIR / 'feature_categories.json', 'w') as f:
    json.dump(feature_categories, f, indent=2)

print(f"   ✓ Saved: feature_categories.json")

# Create feature summary table
feature_summary = pd.DataFrame({
    'Category': list(feature_categories.keys()),
    'N_Features': [len(feature_categories[k]) for k in feature_categories.keys()]
})
feature_summary.to_csv(TABLES_DIR / '09_feature_engineering_summary.csv', index=False)
print(f"   ✓ Saved: 09_feature_engineering_summary.csv")

# Quick statistics on engineered features
print("\n4. Engineered features statistics (train set):")
print("-" * 80)
key_engineered = ['BMI', 'RWT', 'LVMI', 'Fractional_Shortening', 'CV_risk_score', 
                  'cumulative_cardiotoxic_treatment', 'n_CV_risk_factors']
stats = train_eng[key_engineered].describe().T[['mean', 'std', 'min', 'max']]
print(stats.to_string())

print("\n" + "="*80)
print("FEATURE ENGINEERING COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nFeature Summary:")
print(f"  Original features: {len(original_features)}")
print(f"  Engineered features: {len(engineered_features)}")
print(f"  Total features: {len(all_features)}")
print(f"\nFeature Categories:")
for cat, feats in feature_categories.items():
    if cat not in ['original', 'engineered', 'all']:
        print(f"  - {cat}: {len(feats)} features")
print(f"\nReady for feature selection (Phase 4)")
print("="*80)
