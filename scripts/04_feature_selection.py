"""
Phase 4: Feature Selection
Comprehensive feature selection using multiple methods

Methods:
1. Correlation analysis and multicollinearity check (VIF)
2. Univariate statistical tests
3. Tree-based feature importance (XGBoost baseline)
4. Recursive Feature Elimination
5. SHAP-based selection

Author: Competition Submission
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif, RFE
from sklearn.preprocessing import StandardScaler
from statsmodels.stats.outliers_influence import variance_inflation_factor
import xgboost as xgb
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
ENGINEERED_DIR = BASE_DIR / 'data' / 'engineered'
RESULTS_DIR = BASE_DIR / 'results' / 'figures'
TABLES_DIR = BASE_DIR / 'results' / 'tables'

print("="*80)
print("PHASE 4: FEATURE SELECTION")
print("="*80)

# Load data
print("\n1. Loading engineered data...")
train_df = pd.read_csv(ENGINEERED_DIR / 'train_engineered.csv')
val_df = pd.read_csv(ENGINEERED_DIR / 'val_engineered.csv')

X_train = train_df.drop(['CTRCD', 'time'], axis=1)
y_train = train_df['CTRCD']
X_val = val_df.drop(['CTRCD', 'time'], axis=1)
y_val = val_df['CTRCD']

print(f"   Training features: {X_train.shape}")
print(f"   Training target distribution: {y_train.value_counts().to_dict()}")

# ============================================================================
# 1. CORRELATION ANALYSIS
# ============================================================================
print("\n2. Correlation Analysis...")
print("-" * 80)

# Compute correlation matrix
corr_matrix = X_train.corr()

# Find highly correlated features (>0.90)
high_corr_pairs = []
for i in range(len(corr_matrix.columns)):
    for j in range(i+1, len(corr_matrix.columns)):
        if abs(corr_matrix.iloc[i, j]) > 0.90:
            high_corr_pairs.append({
                'Feature_1': corr_matrix.columns[i],
                'Feature_2': corr_matrix.columns[j],
                'Correlation': corr_matrix.iloc[i, j]
            })

if high_corr_pairs:
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Correlation', key=abs, ascending=False)
    print(f"\n   Found {len(high_corr_pairs)} highly correlated pairs (|r| > 0.90):")
    print(high_corr_df.head(10).to_string(index=False))
    high_corr_df.to_csv(TABLES_DIR / '10_high_correlation_pairs.csv', index=False)
    print(f"   ✓ Saved: 10_high_correlation_pairs.csv")
else:
    print("   No feature pairs with correlation > 0.90")

# Correlation with target
target_corr = []
for col in X_train.columns:
    corr = X_train[col].corr(y_train)
    target_corr.append({'Feature': col, 'Correlation': corr, 'Abs_Correlation': abs(corr)})

target_corr_df = pd.DataFrame(target_corr).sort_values('Abs_Correlation', ascending=False)
print(f"\n   Top 15 features by correlation with target:")
print(target_corr_df.head(15).to_string(index=False))
target_corr_df.to_csv(TABLES_DIR / '11_target_correlation.csv', index=False)
print(f"   ✓ Saved: 11_target_correlation.csv")

# Visualize top correlations
fig, ax = plt.subplots(figsize=(12, 8))
top_20 = target_corr_df.head(20).sort_values('Correlation')
colors = ['#e74c3c' if x < 0 else '#2ecc71' for x in top_20['Correlation']]
ax.barh(top_20['Feature'], top_20['Correlation'], color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Correlation with CTRCD', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Top 20 Features by Correlation with Target', fontsize=14, fontweight='bold')
ax.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '11_top_features_correlation.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 11_top_features_correlation.png")

# ============================================================================
# 2. MULTICOLLINEARITY CHECK (VIF)
# ============================================================================
print("\n3. Multicollinearity Check (VIF)...")
print("-" * 80)
print("   Calculating VIF for features (this may take a moment)...")

# Scale features for VIF calculation
scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)

# Calculate VIF for top 30 features (to save time)
top_features = target_corr_df.head(30)['Feature'].tolist()
X_train_vif = X_train_scaled[top_features]

vif_data = []
for i, col in enumerate(X_train_vif.columns):
    try:
        vif = variance_inflation_factor(X_train_vif.values, i)
        vif_data.append({'Feature': col, 'VIF': vif})
    except:
        vif_data.append({'Feature': col, 'VIF': np.nan})

vif_df = pd.DataFrame(vif_data).sort_values('VIF', ascending=False)
print(f"\n   Features with high VIF (>10 indicates multicollinearity):")
print(vif_df[vif_df['VIF'] > 10].to_string(index=False))
vif_df.to_csv(TABLES_DIR / '12_vif_analysis.csv', index=False)
print(f"   ✓ Saved: 12_vif_analysis.csv")

# ============================================================================
# 3. UNIVARIATE STATISTICAL TESTS
# ============================================================================
print("\n4. Univariate Statistical Tests...")
print("-" * 80)

# ANOVA F-test
print("   Running ANOVA F-test...")
f_selector = SelectKBest(score_func=f_classif, k='all')
f_selector.fit(X_train, y_train)

f_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'F_Score': f_selector.scores_,
    'P_Value': f_selector.pvalues_
}).sort_values('F_Score', ascending=False)

print(f"\n   Top 15 features by F-score:")
print(f_scores.head(15).to_string(index=False))
f_scores.to_csv(TABLES_DIR / '13_univariate_f_scores.csv', index=False)
print(f"   ✓ Saved: 13_univariate_f_scores.csv")

# Mutual Information
print("\n   Running Mutual Information test...")
mi_selector = SelectKBest(score_func=mutual_info_classif, k='all')
mi_selector.fit(X_train, y_train)

mi_scores = pd.DataFrame({
    'Feature': X_train.columns,
    'MI_Score': mi_selector.scores_
}).sort_values('MI_Score', ascending=False)

print(f"\n   Top 15 features by Mutual Information:")
print(mi_scores.head(15).to_string(index=False))
mi_scores.to_csv(TABLES_DIR / '14_mutual_information_scores.csv', index=False)
print(f"   ✓ Saved: 14_mutual_information_scores.csv")

# ============================================================================
# 4. TREE-BASED FEATURE IMPORTANCE (XGBoost Baseline)
# ============================================================================
print("\n5. XGBoost Baseline Feature Importance...")
print("-" * 80)

# Calculate scale_pos_weight for imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"   Class imbalance ratio: {scale_pos_weight:.2f}")

# Train baseline XGBoost
print("   Training baseline XGBoost model...")
baseline_model = xgb.XGBClassifier(
    n_estimators=100,
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc',
    tree_method='hist'
)

baseline_model.fit(X_train, y_train)

# Get feature importance
importance_gain = baseline_model.get_booster().get_score(importance_type='gain')
importance_weight = baseline_model.get_booster().get_score(importance_type='weight')
importance_cover = baseline_model.get_booster().get_score(importance_type='cover')

# Create importance dataframe
xgb_importance = pd.DataFrame({
    'Feature': X_train.columns,
    'Gain': [importance_gain.get(f'f{i}', 0) for i in range(len(X_train.columns))],
    'Weight': [importance_weight.get(f'f{i}', 0) for i in range(len(X_train.columns))],
    'Cover': [importance_cover.get(f'f{i}', 0) for i in range(len(X_train.columns))]
}).sort_values('Gain', ascending=False)

print(f"\n   Top 15 features by XGBoost Gain:")
print(xgb_importance.head(15).to_string(index=False))
xgb_importance.to_csv(TABLES_DIR / '15_xgboost_importance.csv', index=False)
print(f"   ✓ Saved: 15_xgboost_importance.csv")

# Visualize XGBoost importance
fig, ax = plt.subplots(figsize=(12, 8))
top_20_xgb = xgb_importance.head(20).sort_values('Gain')
ax.barh(top_20_xgb['Feature'], top_20_xgb['Gain'], color='#3498db', alpha=0.7, edgecolor='black')
ax.set_xlabel('Gain', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Top 20 Features by XGBoost Gain', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '12_xgboost_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 12_xgboost_importance.png")

# ============================================================================
# 5. RECURSIVE FEATURE ELIMINATION (RFE)
# ============================================================================
print("\n6. Recursive Feature Elimination (RFE)...")
print("-" * 80)
print("   Running RFE with XGBoost (selecting top 40 features)...")

rfe_model = xgb.XGBClassifier(
    n_estimators=50,
    max_depth=3,
    learning_rate=0.1,
    scale_pos_weight=scale_pos_weight,
    random_state=42,
    eval_metric='auc'
)

rfe = RFE(estimator=rfe_model, n_features_to_select=40, step=5)
rfe.fit(X_train, y_train)

rfe_results = pd.DataFrame({
    'Feature': X_train.columns,
    'Selected': rfe.support_,
    'Ranking': rfe.ranking_
}).sort_values('Ranking')

print(f"\n   RFE Selected Features (top 40):")
selected_features = rfe_results[rfe_results['Selected']]['Feature'].tolist()
print(f"   {len(selected_features)} features selected")
print(f"   Top 15 by ranking:")
print(rfe_results.head(15).to_string(index=False))

rfe_results.to_csv(TABLES_DIR / '16_rfe_results.csv', index=False)
print(f"   ✓ Saved: 16_rfe_results.csv")

# ============================================================================
# 6. COMPREHENSIVE FEATURE RANKING
# ============================================================================
print("\n7. Creating Comprehensive Feature Ranking...")
print("-" * 80)

# Normalize scores to 0-1 range
def normalize_score(series):
    return (series - series.min()) / (series.max() - series.min() + 1e-10)

# Create comprehensive ranking
comprehensive = pd.DataFrame({'Feature': X_train.columns})

# Add normalized scores
comprehensive = comprehensive.merge(
    target_corr_df[['Feature', 'Abs_Correlation']], on='Feature'
)
comprehensive['Correlation_Score'] = normalize_score(comprehensive['Abs_Correlation'])

comprehensive = comprehensive.merge(
    f_scores[['Feature', 'F_Score']], on='Feature'
)
comprehensive['F_Score_Norm'] = normalize_score(comprehensive['F_Score'])

comprehensive = comprehensive.merge(
    mi_scores[['Feature', 'MI_Score']], on='Feature'
)
comprehensive['MI_Score_Norm'] = normalize_score(comprehensive['MI_Score'])

comprehensive = comprehensive.merge(
    xgb_importance[['Feature', 'Gain']], on='Feature'
)
comprehensive['XGB_Score_Norm'] = normalize_score(comprehensive['Gain'])

comprehensive = comprehensive.merge(
    rfe_results[['Feature', 'Ranking']], on='Feature'
)
comprehensive['RFE_Score_Norm'] = normalize_score(1 / (comprehensive['Ranking'] + 1))

# Calculate composite score (weighted average)
comprehensive['Composite_Score'] = (
    0.20 * comprehensive['Correlation_Score'] +
    0.20 * comprehensive['F_Score_Norm'] +
    0.15 * comprehensive['MI_Score_Norm'] +
    0.30 * comprehensive['XGB_Score_Norm'] +
    0.15 * comprehensive['RFE_Score_Norm']
)

comprehensive = comprehensive.sort_values('Composite_Score', ascending=False)

print(f"\n   Top 20 features by Composite Score:")
print(comprehensive[['Feature', 'Composite_Score', 'Abs_Correlation', 'F_Score', 'Gain']].head(20).to_string(index=False))

comprehensive.to_csv(TABLES_DIR / '17_comprehensive_feature_ranking.csv', index=False)
print(f"   ✓ Saved: 17_comprehensive_feature_ranking.csv")

# Visualize composite scores
fig, ax = plt.subplots(figsize=(12, 10))
top_30 = comprehensive.head(30).sort_values('Composite_Score')
ax.barh(top_30['Feature'], top_30['Composite_Score'], color='#9b59b6', alpha=0.7, edgecolor='black')
ax.set_xlabel('Composite Score', fontsize=12)
ax.set_ylabel('Features', fontsize=12)
ax.set_title('Top 30 Features by Comprehensive Ranking', fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '13_comprehensive_ranking.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 13_comprehensive_ranking.png")

# ============================================================================
# 7. FEATURE SELECTION RECOMMENDATIONS
# ============================================================================
print("\n8. Feature Selection Recommendations...")
print("-" * 80)

# Define feature sets
feature_sets = {
    'top_20': comprehensive.head(20)['Feature'].tolist(),
    'top_30': comprehensive.head(30)['Feature'].tolist(),
    'top_40': comprehensive.head(40)['Feature'].tolist(),
    'top_50': comprehensive.head(50)['Feature'].tolist(),
    'rfe_selected': selected_features,
    'high_importance': xgb_importance[xgb_importance['Gain'] > 0.01]['Feature'].tolist(),
    'significant': f_scores[f_scores['P_Value'] < 0.05]['Feature'].tolist(),
    'all_features': X_train.columns.tolist()
}

import json
with open(ENGINEERED_DIR / 'feature_sets.json', 'w') as f:
    json.dump(feature_sets, f, indent=2)

print(f"\n   Feature Set Recommendations:")
for name, features in feature_sets.items():
    print(f"   - {name}: {len(features)} features")

print(f"   ✓ Saved: feature_sets.json")

# Create feature selection summary
selection_summary = pd.DataFrame({
    'Feature_Set': list(feature_sets.keys()),
    'N_Features': [len(feature_sets[k]) for k in feature_sets.keys()]
})
selection_summary.to_csv(TABLES_DIR / '18_feature_sets_summary.csv', index=False)
print(f"   ✓ Saved: 18_feature_sets_summary.csv")

print("\n" + "="*80)
print("FEATURE SELECTION COMPLETED SUCCESSFULLY")
print("="*80)
print(f"\nKey Findings:")
print(f"  - {len(high_corr_pairs)} highly correlated feature pairs found")
print(f"  - Top feature by composite score: {comprehensive.iloc[0]['Feature']}")
print(f"  - {len(f_scores[f_scores['P_Value'] < 0.05])} statistically significant features (p<0.05)")
print(f"\nRecommended for modeling:")
print(f"  - Start with top_30 or top_40 feature sets")
print(f"  - Compare with RFE selected features")
print(f"  - Iterate based on model performance")
print(f"\nReady for baseline model (Phase 5)")
print("="*80)
