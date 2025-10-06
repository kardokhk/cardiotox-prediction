"""
Phase 7: Comprehensive Hyperparameter Search
Use RandomizedSearchCV to explore broader hyperparameter space

Goals:
1. Define comprehensive parameter grid
2. Implement RandomizedSearchCV with stratified CV
3. Run extensive random search (5000 iterations)
4. Analyze hyperparameter importance
5. Select best hyperparameters based on ROC AUC

Author: Competition Submission
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import xgboost as xgb
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from scipy.stats import uniform, randint
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
ENGINEERED_DIR = BASE_DIR / 'data' / 'engineered'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results' / 'figures'
TABLES_DIR = BASE_DIR / 'results' / 'tables'

print("="*80)
print("PHASE 7: COMPREHENSIVE HYPERPARAMETER SEARCH")
print("="*80)

# Load data
print("\n1. Loading data and previous best configuration...")
train_df = pd.read_csv(ENGINEERED_DIR / 'train_engineered.csv')
val_df = pd.read_csv(ENGINEERED_DIR / 'val_engineered.csv')
test_df = pd.read_csv(ENGINEERED_DIR / 'test_engineered.csv')

# Load features
with open(MODELS_DIR / 'baseline_features.json', 'r') as f:
    features = json.load(f)

# Load previous best metrics
with open(MODELS_DIR / 'optimized_params_iter.json', 'r') as f:
    prev_best_params = json.load(f)

X_train = train_df[features]
y_train = train_df['CTRCD']
X_val = val_df[features]
y_val = val_df['CTRCD']
X_test = test_df[features]
y_test = test_df['CTRCD']

print(f"   Features: {len(features)}")
print(f"   Training samples: {len(X_train)}")
print(f"   Validation samples: {len(X_val)}")
print(f"   Test samples: {len(X_test)}")

# Calculate class imbalance
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos

print(f"   Class imbalance ratio: {scale_pos_weight:.2f}:1")

# ============================================================================
# 2. Define Comprehensive Parameter Distribution
# ============================================================================
print("\n" + "="*80)
print("2. DEFINING COMPREHENSIVE PARAMETER SPACE")
print("="*80)

param_distributions = {
    # Tree structure
    'max_depth': randint(2, 10),
    'min_child_weight': randint(1, 20),
    'max_leaves': randint(0, 50),
    
    # Sampling
    'subsample': uniform(0.5, 0.5),  # 0.5 to 1.0
    'colsample_bytree': uniform(0.5, 0.5),  # 0.5 to 1.0
    'colsample_bylevel': uniform(0.5, 0.5),  # 0.5 to 1.0
    'colsample_bynode': uniform(0.5, 0.5),  # 0.5 to 1.0
    
    # Learning
    'learning_rate': uniform(0.001, 0.299),  # 0.001 to 0.3
    'n_estimators': randint(50, 500),
    
    # Regularization
    'reg_alpha': uniform(0, 10),  # L1 regularization
    'reg_lambda': uniform(0, 10),  # L2 regularization
    'gamma': uniform(0, 10),  # Min loss reduction
    
    # Imbalance handling
    'scale_pos_weight': uniform(1, scale_pos_weight * 2),
    'max_delta_step': randint(0, 10)
}

print("   Parameter distributions defined:")
for param, dist in param_distributions.items():
    print(f"      {param}: {dist}")

# ============================================================================
# 3. Random Search with Cross-Validation
# ============================================================================
print("\n" + "="*80)
print("3. RUNNING RANDOMIZED SEARCH (5000 iterations)")
print("="*80)

# Create stratified CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Base model
base_model = xgb.XGBClassifier(
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

# Randomized search
print("\n   Starting random search (this may take a while)...")
random_search = RandomizedSearchCV(
    estimator=base_model,
    param_distributions=param_distributions,
    n_iter=5000,
    scoring='roc_auc',
    cv=cv,
    verbose=2,
    random_state=42,
    n_jobs=-1,
    return_train_score=True
)

random_search.fit(X_train, y_train)

print("\n   ✓ Random search completed!")

# ============================================================================
# 4. Analyze Results
# ============================================================================
print("\n" + "="*80)
print("4. ANALYZING SEARCH RESULTS")
print("="*80)

# Get results
results_df = pd.DataFrame(random_search.cv_results_)

# Sort by validation score
results_df = results_df.sort_values('rank_test_score')

# Extract best parameters
best_params = random_search.best_params_
best_cv_score = random_search.best_score_

print(f"\n   Best CV ROC AUC: {best_cv_score:.4f}")
print(f"\n   Best parameters:")
for param, value in sorted(best_params.items()):
    print(f"      {param}: {value}")

# Evaluate best model on validation and test sets
best_model = random_search.best_estimator_

val_pred = best_model.predict_proba(X_val)[:, 1]
test_pred = best_model.predict_proba(X_test)[:, 1]
train_pred = best_model.predict_proba(X_train)[:, 1]

train_auc = roc_auc_score(y_train, train_pred)
val_auc = roc_auc_score(y_val, val_pred)
test_auc = roc_auc_score(y_test, test_pred)

train_pr = average_precision_score(y_train, train_pred)
val_pr = average_precision_score(y_val, val_pred)
test_pr = average_precision_score(y_test, test_pred)

print(f"\n   Best Model Performance:")
print(f"      Train ROC AUC: {train_auc:.4f}")
print(f"      Val ROC AUC: {val_auc:.4f}")
print(f"      Test ROC AUC: {test_auc:.4f}")
print(f"\n      Train PR AUC: {train_pr:.4f}")
print(f"      Val PR AUC: {val_pr:.4f}")
print(f"      Test PR AUC: {test_pr:.4f}")

# Save top 20 configurations
top_20 = results_df.head(20)[[
    'rank_test_score', 'mean_test_score', 'std_test_score',
    'mean_train_score', 'std_train_score',
    'param_max_depth', 'param_min_child_weight', 'param_learning_rate',
    'param_n_estimators', 'param_subsample', 'param_colsample_bytree',
    'param_reg_alpha', 'param_reg_lambda', 'param_gamma',
    'param_scale_pos_weight', 'param_max_delta_step'
]]

top_20.to_csv(TABLES_DIR / 'random_search_top20.csv', index=False)
print(f"\n   ✓ Top 20 configurations saved to: {TABLES_DIR / 'random_search_top20.csv'}")

# Save all results
results_df.to_csv(TABLES_DIR / 'random_search_all_results.csv', index=False)
print(f"   ✓ All search results saved to: {TABLES_DIR / 'random_search_all_results.csv'}")

# ============================================================================
# 5. Hyperparameter Importance Analysis
# ============================================================================
print("\n" + "="*80)
print("5. HYPERPARAMETER IMPORTANCE ANALYSIS")
print("="*80)

# Calculate correlation between each parameter and CV score
param_importance = {}

for param in param_distributions.keys():
    param_col = f'param_{param}'
    if param_col in results_df.columns:
        # Handle NaN and convert to numeric
        values = pd.to_numeric(results_df[param_col], errors='coerce')
        scores = results_df['mean_test_score']
        
        # Remove NaN
        mask = ~values.isna()
        if mask.sum() > 1:
            corr = np.corrcoef(values[mask], scores[mask])[0, 1]
            param_importance[param] = abs(corr)

# Sort by importance
param_importance = dict(sorted(param_importance.items(), 
                               key=lambda x: x[1], 
                               reverse=True))

print("\n   Parameter importance (correlation with CV score):")
for param, importance in param_importance.items():
    print(f"      {param:<25} {importance:.4f}")

# Save parameter importance
pd.DataFrame(list(param_importance.items()), 
            columns=['parameter', 'importance']).to_csv(
    TABLES_DIR / 'hyperparameter_importance.csv', index=False)
print(f"\n   ✓ Hyperparameter importance saved to: {TABLES_DIR / 'hyperparameter_importance.csv'}")

# ============================================================================
# 6. Save Best Model
# ============================================================================
print("\n" + "="*80)
print("6. SAVING BEST MODEL FROM RANDOM SEARCH")
print("="*80)

# Save model
best_model.save_model(MODELS_DIR / 'optimized_model_random.json')
print(f"   ✓ Best model saved to: {MODELS_DIR / 'optimized_model_random.json'}")

# Save parameters
with open(MODELS_DIR / 'optimized_params_random.json', 'w') as f:
    # Convert to native Python types
    params_to_save = {}
    for k, v in best_params.items():
        if isinstance(v, (np.integer, np.floating)):
            params_to_save[k] = float(v)
        else:
            params_to_save[k] = v
    json.dump(params_to_save, f, indent=2)
print(f"   ✓ Best parameters saved to: {MODELS_DIR / 'optimized_params_random.json'}")

# Save comprehensive metrics
comprehensive_metrics = {
    'search_type': 'RandomizedSearchCV',
    'n_iterations': 5000,
    'best_cv_score': float(best_cv_score),
    'best_params': params_to_save,
    'performance': {
        'train': {
            'roc_auc': float(train_auc),
            'pr_auc': float(train_pr)
        },
        'validation': {
            'roc_auc': float(val_auc),
            'pr_auc': float(val_pr)
        },
        'test': {
            'roc_auc': float(test_auc),
            'pr_auc': float(test_pr)
        }
    }
}

with open(MODELS_DIR / 'random_search_metrics.json', 'w') as f:
    json.dump(comprehensive_metrics, f, indent=2)
print(f"   ✓ Comprehensive metrics saved to: {MODELS_DIR / 'random_search_metrics.json'}")

# ============================================================================
# 7. Visualizations
# ============================================================================
print("\n" + "="*80)
print("7. CREATING VISUALIZATIONS")
print("="*80)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 7.1 Parameter importance plot
fig, ax = plt.subplots(figsize=(10, 8))

params_list = list(param_importance.keys())[:15]  # Top 15
importance_values = [param_importance[p] for p in params_list]

ax.barh(range(len(params_list)), importance_values, alpha=0.8)
ax.set_yticks(range(len(params_list)))
ax.set_yticklabels(params_list)
ax.invert_yaxis()
ax.set_xlabel('Absolute Correlation with CV ROC AUC', fontsize=11)
ax.set_title('Top 15 Hyperparameter Importance', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '17_hyperparameter_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Hyperparameter importance plot saved to: {RESULTS_DIR / '17_hyperparameter_importance.png'}")

# 7.2 CV score distribution
fig, ax = plt.subplots(figsize=(12, 6))

ax.hist(results_df['mean_test_score'], bins=50, alpha=0.7, edgecolor='black')
ax.axvline(best_cv_score, color='red', linestyle='--', linewidth=2, 
          label=f'Best: {best_cv_score:.4f}')
ax.axvline(results_df['mean_test_score'].median(), color='green', 
          linestyle='--', linewidth=2, 
          label=f'Median: {results_df["mean_test_score"].median():.4f}')

ax.set_xlabel('CV ROC AUC Score', fontsize=11)
ax.set_ylabel('Frequency', fontsize=11)
ax.set_title('Distribution of CV Scores (5000 iterations)', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '18_cv_score_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ CV score distribution plot saved to: {RESULTS_DIR / '18_cv_score_distribution.png'}")

# 7.3 Train vs Test score (overfitting check)
fig, ax = plt.subplots(figsize=(10, 10))

ax.scatter(results_df['mean_train_score'], results_df['mean_test_score'], 
          alpha=0.5, s=30)
ax.plot([0, 1], [0, 1], 'r--', linewidth=2, label='Perfect fit')

# Highlight best model
best_idx = results_df['rank_test_score'].idxmin()
ax.scatter(results_df.loc[best_idx, 'mean_train_score'], 
          results_df.loc[best_idx, 'mean_test_score'],
          color='red', s=200, marker='*', 
          label=f'Best model (CV: {best_cv_score:.4f})', zorder=5)

ax.set_xlabel('Mean Train Score', fontsize=11)
ax.set_ylabel('Mean CV Score', fontsize=11)
ax.set_title('Train vs CV Score - Overfitting Analysis', fontsize=13, fontweight='bold')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '19_train_vs_cv_score.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Train vs CV score plot saved to: {RESULTS_DIR / '19_train_vs_cv_score.png'}")

# 7.4 Top parameters distributions
fig, axes = plt.subplots(2, 3, figsize=(18, 12))
axes = axes.ravel()

top_params = list(param_importance.keys())[:6]

for idx, param in enumerate(top_params):
    param_col = f'param_{param}'
    values = pd.to_numeric(results_df[param_col], errors='coerce')
    scores = results_df['mean_test_score']
    
    # Remove NaN
    mask = ~values.isna()
    
    axes[idx].scatter(values[mask], scores[mask], alpha=0.5, s=20)
    
    # Highlight best
    best_val = best_params[param]
    axes[idx].axvline(best_val, color='red', linestyle='--', linewidth=2,
                     label=f'Best: {best_val:.3f}')
    
    axes[idx].set_xlabel(param, fontsize=10)
    axes[idx].set_ylabel('CV ROC AUC', fontsize=10)
    axes[idx].set_title(f'{param} vs Performance', fontsize=11, fontweight='bold')
    axes[idx].legend(fontsize=8)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '20_top_params_vs_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Top parameters vs performance plot saved to: {RESULTS_DIR / '20_top_params_vs_performance.png'}")

# ============================================================================
# 8. Comparison with Previous Models
# ============================================================================
print("\n" + "="*80)
print("8. COMPARISON WITH PREVIOUS MODELS")
print("="*80)

# Load previous metrics
with open(MODELS_DIR / 'baseline_metrics.json', 'r') as f:
    baseline_metrics = json.load(f)

# Create comparison
comparison = pd.DataFrame({
    'Model': ['Baseline', 'Iterative Optimization', 'Random Search'],
    'Train ROC AUC': [
        baseline_metrics['metrics']['train']['roc_auc'],
        0.7688,  # From phase 6
        train_auc
    ],
    'Val ROC AUC': [
        baseline_metrics['metrics']['validation']['roc_auc'],
        0.6962,  # From phase 6
        val_auc
    ],
    'Test ROC AUC': [
        baseline_metrics['metrics']['test']['roc_auc'],
        np.nan,  # Not evaluated in phase 6
        test_auc
    ],
    'Val PR AUC': [
        baseline_metrics['metrics']['validation']['pr_auc'],
        0.2561,  # From phase 6
        val_pr
    ]
})

print("\n   Model Performance Comparison:")
print(comparison.to_string(index=False))

comparison.to_csv(TABLES_DIR / 'model_comparison.csv', index=False)
print(f"\n   ✓ Model comparison saved to: {TABLES_DIR / 'model_comparison.csv'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 7 COMPLETE: COMPREHENSIVE HYPERPARAMETER SEARCH")
print("="*80)

print(f"\n✓ Completed randomized search with 5000 iterations")
print(f"✓ Best CV ROC AUC: {best_cv_score:.4f}")
print(f"✓ Validation ROC AUC: {val_auc:.4f}")
print(f"✓ Test ROC AUC: {test_auc:.4f}")

print(f"\n✓ Performance improvements:")
baseline_val = baseline_metrics['metrics']['validation']['roc_auc']
print(f"   Baseline → Random Search: {val_auc - baseline_val:+.4f} ({(val_auc - baseline_val)/baseline_val*100:+.2f}%)")

print(f"\n✓ Files saved:")
print(f"   - Best model: {MODELS_DIR / 'optimized_model_random.json'}")
print(f"   - Best parameters: {MODELS_DIR / 'optimized_params_random.json'}")
print(f"   - Search metrics: {MODELS_DIR / 'random_search_metrics.json'}")
print(f"   - Top 20 configs: {TABLES_DIR / 'random_search_top20.csv'}")
print(f"   - All results: {TABLES_DIR / 'random_search_all_results.csv'}")
print(f"   - Hyperparameter importance: {TABLES_DIR / 'hyperparameter_importance.csv'}")
print(f"   - Model comparison: {TABLES_DIR / 'model_comparison.csv'}")
print(f"   - Visualizations: 4 figures saved to {RESULTS_DIR}")

print("\n" + "="*80)
print("Ready for Phase 8: Advanced Techniques (SMOTE, Sampling)")
print("="*80)
