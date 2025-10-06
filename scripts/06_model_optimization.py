"""
Phase 6: Model Optimization - Iterative Approach
Systematically optimize XGBoost hyperparameters in stages

Iterations:
1. Tree structure parameters (max_depth, min_child_weight)
2. Sampling parameters (subsample, colsample_bytree)
3. Learning parameters (learning_rate, n_estimators)
4. Regularization (reg_alpha, reg_lambda, gamma)
5. Imbalance handling (scale_pos_weight, max_delta_step)

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
from sklearn.model_selection import GridSearchCV, StratifiedKFold
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    make_scorer
)
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
ENGINEERED_DIR = BASE_DIR / 'data' / 'engineered'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results' / 'figures'
TABLES_DIR = BASE_DIR / 'results' / 'tables'

print("="*80)
print("PHASE 6: MODEL OPTIMIZATION - ITERATIVE APPROACH")
print("="*80)

# Load data
print("\n1. Loading data and baseline configuration...")
train_df = pd.read_csv(ENGINEERED_DIR / 'train_engineered.csv')
val_df = pd.read_csv(ENGINEERED_DIR / 'val_engineered.csv')

# Load baseline configuration
with open(MODELS_DIR / 'baseline_features.json', 'r') as f:
    features = json.load(f)

with open(MODELS_DIR / 'baseline_metrics.json', 'r') as f:
    baseline_metrics = json.load(f)

X_train = train_df[features]
y_train = train_df['CTRCD']
X_val = val_df[features]
y_val = val_df['CTRCD']

print(f"   Features: {len(features)}")
print(f"   Training samples: {len(X_train)}")
print(f"   Validation samples: {len(X_val)}")
print(f"   Baseline validation ROC AUC: {baseline_metrics['metrics']['validation']['roc_auc']:.4f}")

# Calculate class imbalance
n_neg = (y_train == 0).sum()
n_pos = (y_train == 1).sum()
scale_pos_weight = n_neg / n_pos

# Create stratified CV for hyperparameter tuning
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Track results
optimization_results = []

# Base parameters (from baseline best model)
base_params = {
    'objective': 'binary:logistic',
    'eval_metric': 'auc',
    'random_state': 42,
    'n_jobs': -1
}

# ============================================================================
# ITERATION 1: Optimize Tree Structure Parameters
# ============================================================================
print("\n" + "="*80)
print("ITERATION 1: Optimizing Tree Structure Parameters")
print("="*80)

print("\n   Tuning: max_depth, min_child_weight")

param_grid_1 = {
    'max_depth': [3, 4, 5, 6, 7, 8],
    'min_child_weight': [1, 3, 5, 7, 10]
}

model_1 = xgb.XGBClassifier(**base_params)

grid_search_1 = GridSearchCV(
    estimator=model_1,
    param_grid=param_grid_1,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

grid_search_1.fit(X_train, y_train)

best_params_1 = grid_search_1.best_params_
best_cv_score_1 = grid_search_1.best_score_

# Evaluate on validation set
val_pred_1 = grid_search_1.best_estimator_.predict_proba(X_val)[:, 1]
val_auc_1 = roc_auc_score(y_val, val_pred_1)
val_pr_1 = average_precision_score(y_val, val_pred_1)

print(f"\n   Best parameters: {best_params_1}")
print(f"   Best CV ROC AUC: {best_cv_score_1:.4f}")
print(f"   Validation ROC AUC: {val_auc_1:.4f}")
print(f"   Validation PR AUC: {val_pr_1:.4f}")

optimization_results.append({
    'iteration': 1,
    'stage': 'tree_structure',
    'params_tuned': list(param_grid_1.keys()),
    'best_params': best_params_1,
    'cv_roc_auc': best_cv_score_1,
    'val_roc_auc': val_auc_1,
    'val_pr_auc': val_pr_1
})

# Update base params
base_params.update(best_params_1)

# ============================================================================
# ITERATION 2: Optimize Sampling Parameters
# ============================================================================
print("\n" + "="*80)
print("ITERATION 2: Optimizing Sampling Parameters")
print("="*80)

print("\n   Tuning: subsample, colsample_bytree")

param_grid_2 = {
    'subsample': [0.6, 0.7, 0.8, 0.9, 1.0],
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9, 1.0]
}

model_2 = xgb.XGBClassifier(**base_params)

grid_search_2 = GridSearchCV(
    estimator=model_2,
    param_grid=param_grid_2,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

grid_search_2.fit(X_train, y_train)

best_params_2 = grid_search_2.best_params_
best_cv_score_2 = grid_search_2.best_score_

val_pred_2 = grid_search_2.best_estimator_.predict_proba(X_val)[:, 1]
val_auc_2 = roc_auc_score(y_val, val_pred_2)
val_pr_2 = average_precision_score(y_val, val_pred_2)

print(f"\n   Best parameters: {best_params_2}")
print(f"   Best CV ROC AUC: {best_cv_score_2:.4f}")
print(f"   Validation ROC AUC: {val_auc_2:.4f}")
print(f"   Validation PR AUC: {val_pr_2:.4f}")

optimization_results.append({
    'iteration': 2,
    'stage': 'sampling',
    'params_tuned': list(param_grid_2.keys()),
    'best_params': best_params_2,
    'cv_roc_auc': best_cv_score_2,
    'val_roc_auc': val_auc_2,
    'val_pr_auc': val_pr_2
})

base_params.update(best_params_2)

# ============================================================================
# ITERATION 3: Optimize Learning Parameters
# ============================================================================
print("\n" + "="*80)
print("ITERATION 3: Optimizing Learning Parameters")
print("="*80)

print("\n   Tuning: learning_rate, n_estimators")

param_grid_3 = {
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'n_estimators': [100, 200, 300, 500]
}

model_3 = xgb.XGBClassifier(**base_params)

grid_search_3 = GridSearchCV(
    estimator=model_3,
    param_grid=param_grid_3,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

grid_search_3.fit(X_train, y_train)

best_params_3 = grid_search_3.best_params_
best_cv_score_3 = grid_search_3.best_score_

val_pred_3 = grid_search_3.best_estimator_.predict_proba(X_val)[:, 1]
val_auc_3 = roc_auc_score(y_val, val_pred_3)
val_pr_3 = average_precision_score(y_val, val_pred_3)

print(f"\n   Best parameters: {best_params_3}")
print(f"   Best CV ROC AUC: {best_cv_score_3:.4f}")
print(f"   Validation ROC AUC: {val_auc_3:.4f}")
print(f"   Validation PR AUC: {val_pr_3:.4f}")

optimization_results.append({
    'iteration': 3,
    'stage': 'learning',
    'params_tuned': list(param_grid_3.keys()),
    'best_params': best_params_3,
    'cv_roc_auc': best_cv_score_3,
    'val_roc_auc': val_auc_3,
    'val_pr_auc': val_pr_3
})

base_params.update(best_params_3)

# ============================================================================
# ITERATION 4: Optimize Regularization Parameters
# ============================================================================
print("\n" + "="*80)
print("ITERATION 4: Optimizing Regularization Parameters")
print("="*80)

print("\n   Tuning: reg_alpha, reg_lambda, gamma")

param_grid_4 = {
    'reg_alpha': [0, 0.01, 0.1, 1, 10],
    'reg_lambda': [0, 0.01, 0.1, 1, 10],
    'gamma': [0, 0.1, 0.5, 1, 5]
}

model_4 = xgb.XGBClassifier(**base_params)

grid_search_4 = GridSearchCV(
    estimator=model_4,
    param_grid=param_grid_4,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

grid_search_4.fit(X_train, y_train)

best_params_4 = grid_search_4.best_params_
best_cv_score_4 = grid_search_4.best_score_

val_pred_4 = grid_search_4.best_estimator_.predict_proba(X_val)[:, 1]
val_auc_4 = roc_auc_score(y_val, val_pred_4)
val_pr_4 = average_precision_score(y_val, val_pred_4)

print(f"\n   Best parameters: {best_params_4}")
print(f"   Best CV ROC AUC: {best_cv_score_4:.4f}")
print(f"   Validation ROC AUC: {val_auc_4:.4f}")
print(f"   Validation PR AUC: {val_pr_4:.4f}")

optimization_results.append({
    'iteration': 4,
    'stage': 'regularization',
    'params_tuned': list(param_grid_4.keys()),
    'best_params': best_params_4,
    'cv_roc_auc': best_cv_score_4,
    'val_roc_auc': val_auc_4,
    'val_pr_auc': val_pr_4
})

base_params.update(best_params_4)

# ============================================================================
# ITERATION 5: Optimize Imbalance Handling
# ============================================================================
print("\n" + "="*80)
print("ITERATION 5: Optimizing Imbalance Handling")
print("="*80)

print("\n   Tuning: scale_pos_weight, max_delta_step")

param_grid_5 = {
    'scale_pos_weight': [1, scale_pos_weight/2, scale_pos_weight, scale_pos_weight*1.5, scale_pos_weight*2],
    'max_delta_step': [0, 1, 3, 5, 10]
}

model_5 = xgb.XGBClassifier(**base_params)

grid_search_5 = GridSearchCV(
    estimator=model_5,
    param_grid=param_grid_5,
    scoring='roc_auc',
    cv=cv,
    verbose=1,
    n_jobs=-1
)

grid_search_5.fit(X_train, y_train)

best_params_5 = grid_search_5.best_params_
best_cv_score_5 = grid_search_5.best_score_

val_pred_5 = grid_search_5.best_estimator_.predict_proba(X_val)[:, 1]
val_auc_5 = roc_auc_score(y_val, val_pred_5)
val_pr_5 = average_precision_score(y_val, val_pred_5)

print(f"\n   Best parameters: {best_params_5}")
print(f"   Best CV ROC AUC: {best_cv_score_5:.4f}")
print(f"   Validation ROC AUC: {val_auc_5:.4f}")
print(f"   Validation PR AUC: {val_pr_5:.4f}")

optimization_results.append({
    'iteration': 5,
    'stage': 'imbalance_handling',
    'params_tuned': list(param_grid_5.keys()),
    'best_params': best_params_5,
    'cv_roc_auc': best_cv_score_5,
    'val_roc_auc': val_auc_5,
    'val_pr_auc': val_pr_5
})

base_params.update(best_params_5)

# ============================================================================
# Save Optimization Results
# ============================================================================
print("\n" + "="*80)
print("OPTIMIZATION SUMMARY")
print("="*80)

# Create results DataFrame
results_df = pd.DataFrame(optimization_results)

print("\n   Optimization Progress:")
print(results_df[['iteration', 'stage', 'cv_roc_auc', 'val_roc_auc', 'val_pr_auc']].to_string(index=False))

# Save results
results_df.to_csv(TABLES_DIR / 'optimization_iterations.csv', index=False)
print(f"\n   ✓ Optimization results saved to: {TABLES_DIR / 'optimization_iterations.csv'}")

# Save full results with all parameters
with open(TABLES_DIR / 'optimization_iterations_detailed.json', 'w') as f:
    json.dump(optimization_results, f, indent=2)
print(f"   ✓ Detailed results saved to: {TABLES_DIR / 'optimization_iterations_detailed.json'}")

# ============================================================================
# Train Final Optimized Model
# ============================================================================
print("\n" + "="*80)
print("TRAINING FINAL OPTIMIZED MODEL")
print("="*80)

# Remove keys that are not XGBoost parameters
final_params = {k: v for k, v in base_params.items() if k not in ['n_jobs']}
final_params['n_jobs'] = -1

print("\n   Final optimized parameters:")
for key, value in sorted(final_params.items()):
    print(f"      {key}: {value}")

# Train final model
final_model = xgb.XGBClassifier(**final_params)
final_model.fit(X_train, y_train)

# Evaluate on train and validation
train_pred = final_model.predict_proba(X_train)[:, 1]
val_pred = final_model.predict_proba(X_val)[:, 1]

train_auc = roc_auc_score(y_train, train_pred)
val_auc = roc_auc_score(y_val, val_pred)
train_pr = average_precision_score(y_train, train_pred)
val_pr = average_precision_score(y_val, val_pred)

print("\n   Final Model Performance:")
print(f"      Train ROC AUC: {train_auc:.4f} | Validation ROC AUC: {val_auc:.4f}")
print(f"      Train PR AUC:  {train_pr:.4f} | Validation PR AUC:  {val_pr:.4f}")

print(f"\n   Improvement over baseline:")
baseline_val_auc = baseline_metrics['metrics']['validation']['roc_auc']
improvement = val_auc - baseline_val_auc
print(f"      ROC AUC improvement: {improvement:+.4f} ({improvement/baseline_val_auc*100:+.2f}%)")

# Save optimized model
final_model.save_model(MODELS_DIR / 'optimized_model_iter.json')
print(f"\n   ✓ Optimized model saved to: {MODELS_DIR / 'optimized_model_iter.json'}")

# Save optimized parameters
with open(MODELS_DIR / 'optimized_params_iter.json', 'w') as f:
    # Convert numpy types to native Python types
    params_to_save = {}
    for k, v in final_params.items():
        if isinstance(v, (np.integer, np.floating)):
            params_to_save[k] = float(v)
        else:
            params_to_save[k] = v
    json.dump(params_to_save, f, indent=2)
print(f"   ✓ Optimized parameters saved to: {MODELS_DIR / 'optimized_params_iter.json'}")

# ============================================================================
# Visualizations
# ============================================================================
print("\n" + "="*80)
print("CREATING OPTIMIZATION VISUALIZATIONS")
print("="*80)

plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 1. Optimization Progress Plot
fig, axes = plt.subplots(1, 2, figsize=(15, 5))

# CV ROC AUC progress
axes[0].plot(results_df['iteration'], results_df['cv_roc_auc'], 
             marker='o', linewidth=2, markersize=8, label='CV ROC AUC')
axes[0].axhline(y=baseline_val_auc, color='red', linestyle='--', 
                linewidth=2, label=f'Baseline ({baseline_val_auc:.4f})')
axes[0].set_xlabel('Iteration', fontsize=11)
axes[0].set_ylabel('ROC AUC', fontsize=11)
axes[0].set_title('Cross-Validation Performance Progress', fontsize=13, fontweight='bold')
axes[0].legend(fontsize=10)
axes[0].grid(True, alpha=0.3)
axes[0].set_xticks(results_df['iteration'])

# Validation ROC AUC progress
axes[1].plot(results_df['iteration'], results_df['val_roc_auc'], 
             marker='s', linewidth=2, markersize=8, label='Validation ROC AUC', color='green')
axes[1].axhline(y=baseline_val_auc, color='red', linestyle='--', 
                linewidth=2, label=f'Baseline ({baseline_val_auc:.4f})')
axes[1].set_xlabel('Iteration', fontsize=11)
axes[1].set_ylabel('ROC AUC', fontsize=11)
axes[1].set_title('Validation Performance Progress', fontsize=13, fontweight='bold')
axes[1].legend(fontsize=10)
axes[1].grid(True, alpha=0.3)
axes[1].set_xticks(results_df['iteration'])

plt.tight_layout()
plt.savefig(RESULTS_DIR / '15_optimization_progress.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Optimization progress plot saved to: {RESULTS_DIR / '15_optimization_progress.png'}")

# 2. Stage-wise improvement
fig, ax = plt.subplots(figsize=(12, 6))

x = np.arange(len(results_df))
width = 0.35

bars1 = ax.bar(x - width/2, results_df['cv_roc_auc'], width, 
               label='CV ROC AUC', alpha=0.8)
bars2 = ax.bar(x + width/2, results_df['val_roc_auc'], width, 
               label='Validation ROC AUC', alpha=0.8)

ax.axhline(y=baseline_val_auc, color='red', linestyle='--', 
           linewidth=2, label=f'Baseline ({baseline_val_auc:.4f})')

ax.set_xlabel('Optimization Stage', fontsize=11)
ax.set_ylabel('ROC AUC', fontsize=11)
ax.set_title('Performance by Optimization Stage', fontsize=13, fontweight='bold')
ax.set_xticks(x)
ax.set_xticklabels([f"Iter {i+1}\n{stage}" for i, stage in 
                    enumerate(results_df['stage'])], rotation=15, ha='right')
ax.legend(fontsize=10)
ax.grid(True, alpha=0.3, axis='y')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '16_optimization_stages.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Optimization stages plot saved to: {RESULTS_DIR / '16_optimization_stages.png'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 6 COMPLETE: MODEL OPTIMIZATION - ITERATIVE APPROACH")
print("="*80)

print(f"\n✓ Completed 5 optimization iterations")
print(f"✓ Final optimized model trained and saved")
print(f"\n✓ Performance comparison:")
print(f"   Baseline validation ROC AUC: {baseline_val_auc:.4f}")
print(f"   Optimized validation ROC AUC: {val_auc:.4f}")
print(f"   Improvement: {improvement:+.4f} ({improvement/baseline_val_auc*100:+.2f}%)")

print(f"\n✓ Files saved:")
print(f"   - Optimized model: {MODELS_DIR / 'optimized_model_iter.json'}")
print(f"   - Optimized parameters: {MODELS_DIR / 'optimized_params_iter.json'}")
print(f"   - Iteration results: {TABLES_DIR / 'optimization_iterations.csv'}")
print(f"   - Detailed results: {TABLES_DIR / 'optimization_iterations_detailed.json'}")
print(f"   - Visualizations: 2 figures saved to {RESULTS_DIR}")

print("\n" + "="*80)
print("Ready for Phase 7: Comprehensive Hyperparameter Search")
print("="*80)
