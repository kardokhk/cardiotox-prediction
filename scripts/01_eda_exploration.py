"""
Phase 1: Exploratory Data Analysis (EDA)
Comprehensive analysis of the cardiotoxicity dataset

Author: Kardokh Kaka Bra
Date: October 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 10

# Paths
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / 'dataset'
RESULTS_DIR = BASE_DIR / 'results' / 'figures'
TABLES_DIR = BASE_DIR / 'results' / 'tables'

# Ensure directories exist
RESULTS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("CARDIOTOXICITY PREDICTION - EXPLORATORY DATA ANALYSIS")
print("="*80)

# Load data with proper delimiter and decimal
print("\n1. Loading data...")
df = pd.read_csv(DATA_DIR / 'BC_cardiotox_clinical_variables.csv', 
                 sep=';', decimal=',')

print(f"   Dataset shape: {df.shape}")
print(f"   Patients: {df.shape[0]}")
print(f"   Features: {df.shape[1]}")

# Basic information
print("\n2. Dataset Overview:")
print("-" * 80)
print(df.info())

# Target variable analysis
print("\n3. Target Variable (CTRCD) Analysis:")
print("-" * 80)
ctrcd_counts = df['CTRCD'].value_counts()
print(f"   No CTRCD (0): {ctrcd_counts[0]} ({ctrcd_counts[0]/len(df)*100:.2f}%)")
print(f"   CTRCD (1): {ctrcd_counts[1]} ({ctrcd_counts[1]/len(df)*100:.2f}%)")
print(f"   Imbalance Ratio: {ctrcd_counts[0]/ctrcd_counts[1]:.2f}:1")

# Save target distribution plot
fig, ax = plt.subplots(figsize=(8, 6))
colors = ['#2ecc71', '#e74c3c']
bars = ax.bar(['No CTRCD', 'CTRCD'], ctrcd_counts.values, color=colors, alpha=0.7, edgecolor='black')
ax.set_ylabel('Number of Patients', fontsize=12)
ax.set_title('Target Variable Distribution: CTRCD', fontsize=14, fontweight='bold')
# Add value labels on bars
for bar in bars:
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
            f'{int(height)}\n({height/len(df)*100:.1f}%)',
            ha='center', va='bottom', fontsize=11)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '01_target_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 01_target_distribution.png")

# Missing values analysis
print("\n4. Missing Values Analysis:")
print("-" * 80)
missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_df = pd.DataFrame({
    'Feature': missing.index,
    'Missing_Count': missing.values,
    'Missing_Percentage': missing_pct.values
}).sort_values('Missing_Count', ascending=False)

print(missing_df[missing_df['Missing_Count'] > 0].to_string(index=False))

# Save missing values table
missing_df[missing_df['Missing_Count'] > 0].to_csv(
    TABLES_DIR / '01_missing_values_summary.csv', index=False
)
print(f"   ✓ Saved: 01_missing_values_summary.csv")

# Visualize missing data
fig, ax = plt.subplots(figsize=(10, 8))
missing_features = missing_df[missing_df['Missing_Count'] > 0].sort_values('Missing_Count')
ax.barh(missing_features['Feature'], missing_features['Missing_Percentage'], 
        color='#e74c3c', alpha=0.7, edgecolor='black')
ax.set_xlabel('Missing Percentage (%)', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Missing Values by Feature', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '02_missing_values.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 02_missing_values.png")

# Descriptive statistics
print("\n5. Descriptive Statistics:")
print("-" * 80)

# Separate continuous and binary variables
continuous_vars = ['age', 'weight', 'height', 'heart_rate', 'LVEF', 'PWT', 'LAd', 'LVDd', 'LVSd', 'time']
binary_vars = [col for col in df.columns if col not in continuous_vars + ['CTRCD']]

# Continuous variables
desc_stats = df[continuous_vars].describe().T
desc_stats['missing'] = df[continuous_vars].isnull().sum()
desc_stats['missing_pct'] = (desc_stats['missing'] / len(df) * 100).round(2)
print("\nContinuous Variables:")
print(desc_stats.to_string())
desc_stats.to_csv(TABLES_DIR / '02_continuous_variables_stats.csv')
print(f"   ✓ Saved: 02_continuous_variables_stats.csv")

# Binary variables
print("\nBinary Variables Distribution:")
binary_summary = []
for var in binary_vars:
    if var in df.columns:
        counts = df[var].value_counts()
        total = counts.sum()
        binary_summary.append({
            'Variable': var,
            'Value_0_Count': counts.get(0.0, 0),
            'Value_0_Pct': f"{counts.get(0.0, 0)/total*100:.2f}%" if total > 0 else "N/A",
            'Value_1_Count': counts.get(1.0, 0),
            'Value_1_Pct': f"{counts.get(1.0, 0)/total*100:.2f}%" if total > 0 else "N/A",
            'Missing': df[var].isnull().sum()
        })
binary_summary_df = pd.DataFrame(binary_summary)
print(binary_summary_df.to_string(index=False))
binary_summary_df.to_csv(TABLES_DIR / '03_binary_variables_stats.csv', index=False)
print(f"   ✓ Saved: 03_binary_variables_stats.csv")

# Distribution plots for continuous variables
print("\n6. Creating distribution plots for continuous variables...")
for var in continuous_vars:
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Histogram
    df[var].hist(bins=30, ax=axes[0], color='#3498db', alpha=0.7, edgecolor='black')
    axes[0].set_xlabel(var, fontsize=12)
    axes[0].set_ylabel('Frequency', fontsize=12)
    axes[0].set_title(f'Distribution of {var}', fontsize=13, fontweight='bold')
    axes[0].grid(axis='y', alpha=0.3)
    
    # Box plot by CTRCD
    df.boxplot(column=var, by='CTRCD', ax=axes[1])
    axes[1].set_xlabel('CTRCD Status', fontsize=12)
    axes[1].set_ylabel(var, fontsize=12)
    axes[1].set_title(f'{var} by CTRCD Status', fontsize=13, fontweight='bold')
    axes[1].set_xticklabels(['No CTRCD', 'CTRCD'])
    plt.suptitle('')  # Remove default title
    
    plt.tight_layout()
    plt.savefig(RESULTS_DIR / f'03_dist_{var}.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"   ✓ Saved: 03_dist_{var}.png")

# Correlation analysis
print("\n7. Correlation Analysis...")
# Select numeric columns for correlation
numeric_cols = df[continuous_vars].columns
corr_matrix = df[numeric_cols].corr()

# Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='coolwarm', 
            center=0, square=True, linewidths=1, cbar_kws={"shrink": 0.8},
            ax=ax)
ax.set_title('Correlation Matrix - Continuous Variables', fontsize=14, fontweight='bold', pad=20)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '04_correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 04_correlation_matrix.png")

# Correlation with target
print("\n8. Correlation with Target Variable (CTRCD)...")
# Calculate point-biserial correlation for continuous variables
target_corr = []
for var in continuous_vars:
    valid_data = df[[var, 'CTRCD']].dropna()
    if len(valid_data) > 0:
        corr = valid_data[var].corr(valid_data['CTRCD'])
        target_corr.append({'Variable': var, 'Correlation': corr})

target_corr_df = pd.DataFrame(target_corr).sort_values('Correlation', key=abs, ascending=False)
print(target_corr_df.to_string(index=False))
target_corr_df.to_csv(TABLES_DIR / '04_target_correlation.csv', index=False)
print(f"   ✓ Saved: 04_target_correlation.csv")

# Visualize correlation with target
fig, ax = plt.subplots(figsize=(10, 8))
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in target_corr_df['Correlation']]
ax.barh(target_corr_df['Variable'], target_corr_df['Correlation'], 
        color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Correlation with CTRCD', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Feature Correlation with Target (CTRCD)', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '05_target_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 05_target_correlation.png")

# Binary variables vs CTRCD
print("\n9. Analyzing binary variables vs CTRCD...")
binary_target_analysis = []
for var in binary_vars:
    if var in df.columns:
        crosstab = pd.crosstab(df[var], df['CTRCD'], normalize='index') * 100
        if len(crosstab) > 0:
            # Chi-square test
            from scipy.stats import chi2_contingency
            contingency = pd.crosstab(df[var], df['CTRCD'])
            if contingency.shape[0] > 1 and contingency.shape[1] > 1:
                chi2, p_value, dof, expected = chi2_contingency(contingency)
                binary_target_analysis.append({
                    'Variable': var,
                    'Chi2_Statistic': chi2,
                    'P_Value': p_value,
                    'Significant': 'Yes' if p_value < 0.05 else 'No'
                })

binary_target_df = pd.DataFrame(binary_target_analysis).sort_values('P_Value')
print(binary_target_df.to_string(index=False))
binary_target_df.to_csv(TABLES_DIR / '05_binary_vs_target_chisquare.csv', index=False)
print(f"   ✓ Saved: 05_binary_vs_target_chisquare.csv")

# Outlier detection
print("\n10. Outlier Detection (IQR method)...")
outlier_summary = []
for var in continuous_vars:
    Q1 = df[var].quantile(0.25)
    Q3 = df[var].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    outliers = df[(df[var] < lower_bound) | (df[var] > upper_bound)][var]
    outlier_summary.append({
        'Variable': var,
        'N_Outliers': len(outliers),
        'Outlier_Percentage': f"{len(outliers)/len(df)*100:.2f}%",
        'Lower_Bound': lower_bound,
        'Upper_Bound': upper_bound
    })

outlier_df = pd.DataFrame(outlier_summary)
print(outlier_df.to_string(index=False))
outlier_df.to_csv(TABLES_DIR / '06_outlier_summary.csv', index=False)
print(f"   ✓ Saved: 06_outlier_summary.csv")

# Age distribution by CTRCD
print("\n11. Creating age group analysis...")
df['age_group'] = pd.cut(df['age'], bins=[0, 40, 50, 60, 70, 100], 
                          labels=['<40', '40-50', '50-60', '60-70', '70+'])
age_group_ctrcd = pd.crosstab(df['age_group'], df['CTRCD'], normalize='index') * 100

fig, ax = plt.subplots(figsize=(10, 6))
age_group_ctrcd.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Age Group', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('CTRCD Rate by Age Group', fontsize=14, fontweight='bold')
ax.legend(['No CTRCD', 'CTRCD'], title='CTRCD Status')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45)
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '06_age_group_ctrcd.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 06_age_group_ctrcd.png")

# LVEF distribution analysis (critical variable)
print("\n12. LVEF (Left Ventricular Ejection Fraction) Analysis...")
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# LVEF distribution by CTRCD
for ctrcd in [0, 1]:
    data = df[df['CTRCD'] == ctrcd]['LVEF'].dropna()
    axes[0].hist(data, bins=20, alpha=0.6, label=f'CTRCD={ctrcd}', edgecolor='black')
axes[0].axvline(x=50, color='red', linestyle='--', linewidth=2, label='CTRCD Threshold (50%)')
axes[0].set_xlabel('LVEF (%)', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].set_title('LVEF Distribution by CTRCD Status', fontsize=13, fontweight='bold')
axes[0].legend()
axes[0].grid(axis='y', alpha=0.3)

# LVEF categories
df['LVEF_category'] = pd.cut(df['LVEF'], bins=[0, 50, 55, 60, 100], 
                              labels=['<50 (Low)', '50-55 (Borderline)', '55-60 (Normal)', '>60 (High)'])
lvef_cat_ctrcd = pd.crosstab(df['LVEF_category'], df['CTRCD'], normalize='index') * 100
lvef_cat_ctrcd.plot(kind='bar', ax=axes[1], color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
axes[1].set_xlabel('LVEF Category', fontsize=12)
axes[1].set_ylabel('Percentage (%)', fontsize=12)
axes[1].set_title('CTRCD Rate by LVEF Category', fontsize=13, fontweight='bold')
axes[1].legend(['No CTRCD', 'CTRCD'], title='CTRCD Status')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '07_lvef_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 07_lvef_analysis.png")

# Treatment combinations analysis
print("\n13. Treatment Combinations Analysis...")
treatment_cols = ['AC', 'antiHER2']
df_treatment = df[treatment_cols + ['CTRCD']].dropna()
df_treatment['treatment_combo'] = (
    df_treatment['AC'].astype(str) + '_' + 
    df_treatment['antiHER2'].astype(str)
).replace({
    '0.0_0.0': 'None',
    '0.0_1.0': 'antiHER2 only',
    '1.0_0.0': 'AC only',
    '1.0_1.0': 'Both AC & antiHER2'
})

treatment_ctrcd = pd.crosstab(df_treatment['treatment_combo'], 
                               df_treatment['CTRCD'], normalize='index') * 100

fig, ax = plt.subplots(figsize=(10, 6))
treatment_ctrcd.plot(kind='bar', ax=ax, color=['#2ecc71', '#e74c3c'], alpha=0.7, edgecolor='black')
ax.set_xlabel('Treatment Combination', fontsize=12)
ax.set_ylabel('Percentage (%)', fontsize=12)
ax.set_title('CTRCD Rate by Treatment Combination', fontsize=14, fontweight='bold')
ax.legend(['No CTRCD', 'CTRCD'], title='CTRCD Status')
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '08_treatment_combinations.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 08_treatment_combinations.png")

# Risk factors analysis
print("\n14. Cardiovascular Risk Factors Analysis...")
risk_factors = ['HTA', 'DL', 'DM', 'smoker', 'exsmoker']
risk_df = df[risk_factors + ['CTRCD']].dropna()

risk_summary = []
for risk in risk_factors:
    total = risk_df[risk].sum()
    ctrcd_with_risk = risk_df[risk_df[risk] == 1]['CTRCD'].sum()
    ctrcd_rate = (ctrcd_with_risk / total * 100) if total > 0 else 0
    risk_summary.append({
        'Risk_Factor': risk,
        'Patients_With_Risk': int(total),
        'CTRCD_Cases': int(ctrcd_with_risk),
        'CTRCD_Rate_Percent': f"{ctrcd_rate:.2f}%"
    })

risk_summary_df = pd.DataFrame(risk_summary)
print(risk_summary_df.to_string(index=False))
risk_summary_df.to_csv(TABLES_DIR / '07_risk_factors_summary.csv', index=False)
print(f"   ✓ Saved: 07_risk_factors_summary.csv")

# Risk factors visualization
fig, ax = plt.subplots(figsize=(10, 6))
risk_names = [r.replace('exsmoker', 'Ex-Smoker').replace('smoker', 'Current Smoker').replace('HTA', 'Hypertension').replace('DL', 'Dyslipidemia').replace('DM', 'Diabetes') for r in risk_factors]
risk_rates = [float(r.replace('%', '')) for r in risk_summary_df['CTRCD_Rate_Percent']]
colors_risk = ['#e74c3c' if r > 10 else '#f39c12' if r > 5 else '#3498db' for r in risk_rates]
bars = ax.bar(risk_names, risk_rates, color=colors_risk, alpha=0.7, edgecolor='black')
ax.axhline(y=10.17, color='black', linestyle='--', linewidth=2, label='Overall CTRCD Rate (10.17%)')
ax.set_ylabel('CTRCD Rate (%)', fontsize=12)
ax.set_xlabel('Risk Factors', fontsize=12)
ax.set_title('CTRCD Rate by Cardiovascular Risk Factors', fontsize=14, fontweight='bold')
ax.set_xticklabels(risk_names, rotation=45, ha='right')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '09_risk_factors_ctrcd.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 09_risk_factors_ctrcd.png")

# Time to CTRCD analysis
print("\n15. Time to CTRCD Event Analysis...")
ctrcd_patients = df[df['CTRCD'] == 1]['time']
print(f"   Mean time to CTRCD: {ctrcd_patients.mean():.1f} days ({ctrcd_patients.mean()/30:.1f} months)")
print(f"   Median time to CTRCD: {ctrcd_patients.median():.1f} days ({ctrcd_patients.median()/30:.1f} months)")
print(f"   Min time to CTRCD: {ctrcd_patients.min():.1f} days")
print(f"   Max time to CTRCD: {ctrcd_patients.max():.1f} days")

fig, ax = plt.subplots(figsize=(10, 6))
ax.hist(ctrcd_patients, bins=20, color='#e74c3c', alpha=0.7, edgecolor='black')
ax.axvline(x=ctrcd_patients.mean(), color='blue', linestyle='--', linewidth=2, 
           label=f'Mean: {ctrcd_patients.mean():.0f} days')
ax.axvline(x=ctrcd_patients.median(), color='green', linestyle='--', linewidth=2,
           label=f'Median: {ctrcd_patients.median():.0f} days')
ax.set_xlabel('Time to CTRCD (days)', fontsize=12)
ax.set_ylabel('Frequency', fontsize=12)
ax.set_title('Distribution of Time to CTRCD Event', fontsize=14, fontweight='bold')
ax.legend()
ax.grid(axis='y', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '10_time_to_ctrcd.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 10_time_to_ctrcd.png")

# Final summary
print("\n" + "="*80)
print("EDA COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nGenerated Files:")
print(f"  - Figures: {len(list(RESULTS_DIR.glob('*.png')))} plots in {RESULTS_DIR}")
print(f"  - Tables: {len(list(TABLES_DIR.glob('*.csv')))} CSV files in {TABLES_DIR}")
print("\nKey Findings:")
print(f"  - Dataset: {df.shape[0]} patients, {df.shape[1]} features")
print(f"  - Target: {ctrcd_counts[1]} CTRCD cases ({ctrcd_counts[1]/len(df)*100:.2f}%)")
print(f"  - Imbalance ratio: {ctrcd_counts[0]/ctrcd_counts[1]:.1f}:1")
print(f"  - Missing data: ~50 patients missing treatment/risk data")
print(f"  - Mean LVEF: {df['LVEF'].mean():.1f}%")
print(f"  - Mean age: {df['age'].mean():.1f} years")
print("="*80)

