"""
Phase 5: Baseline Model Development
Create baseline XGBoost model with default parameters and class imbalance handling

Goals:
1. Create baseline XGBoost model with default parameters
2. Implement class imbalance handling (scale_pos_weight)
3. Evaluate baseline performance (ROC AUC, PR AUC, confusion matrix)
4. Establish performance benchmarks

Author: Kardokh Kaka Bra
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
import json
import xgboost as xgb
from sklearn.metrics import (
    roc_auc_score, average_precision_score, 
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve,
    f1_score, accuracy_score
)
import warnings
warnings.filterwarnings('ignore')

# Paths
BASE_DIR = Path(__file__).parent.parent
ENGINEERED_DIR = BASE_DIR / 'data' / 'engineered'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results' / 'figures'
TABLES_DIR = BASE_DIR / 'results' / 'tables'

# Create directories
MODELS_DIR.mkdir(parents=True, exist_ok=True)
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 5: BASELINE MODEL DEVELOPMENT")
print("="*80)

# Load data
print("\n1. Loading engineered data...")
train_df = pd.read_csv(ENGINEERED_DIR / 'train_engineered.csv')
val_df = pd.read_csv(ENGINEERED_DIR / 'val_engineered.csv')
test_df = pd.read_csv(ENGINEERED_DIR / 'test_engineered.csv')

# Load feature sets
with open(ENGINEERED_DIR / 'feature_sets.json', 'r') as f:
    feature_sets = json.load(f)

print(f"   Training samples: {len(train_df)}")
print(f"   Validation samples: {len(val_df)}")
print(f"   Test samples: {len(test_df)}")
print(f"   Training CTRCD distribution: {train_df['CTRCD'].value_counts().to_dict()}")

# Calculate class imbalance
n_neg = (train_df['CTRCD'] == 0).sum()
n_pos = (train_df['CTRCD'] == 1).sum()
scale_pos_weight = n_neg / n_pos
print(f"\n   Class imbalance ratio: {scale_pos_weight:.2f}:1")
print(f"   Recommended scale_pos_weight: {scale_pos_weight:.2f}")

# ============================================================================
# 2. Baseline Model 1: Default XGBoost (No Class Balancing)
# ============================================================================
print("\n" + "="*80)
print("2. BASELINE MODEL 1: Default XGBoost (No Class Balancing)")
print("="*80)

# Test on all feature sets
results = {}

for feature_set_name in ['top_20', 'top_30', 'top_40', 'all_features']:
    print(f"\n   Testing feature set: {feature_set_name}")
    
    # Select features
    if feature_set_name == 'all_features':
        features = [col for col in train_df.columns if col not in ['CTRCD', 'time']]
    else:
        features = feature_sets[feature_set_name]
    
    X_train = train_df[features]
    y_train = train_df['CTRCD']
    X_val = val_df[features]
    y_val = val_df['CTRCD']
    X_test = test_df[features]
    y_test = test_df['CTRCD']
    
    print(f"      Features: {len(features)}")
    
    # Train baseline model (default parameters)
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='auc'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    train_pr = average_precision_score(y_train, y_train_pred)
    val_pr = average_precision_score(y_val, y_val_pred)
    test_pr = average_precision_score(y_test, y_test_pred)
    
    results[f'baseline_default_{feature_set_name}'] = {
        'feature_set': feature_set_name,
        'n_features': len(features),
        'train_roc_auc': train_auc,
        'val_roc_auc': val_auc,
        'test_roc_auc': test_auc,
        'train_pr_auc': train_pr,
        'val_pr_auc': val_pr,
        'test_pr_auc': test_pr,
        'scale_pos_weight': None,
        'model_type': 'default'
    }
    
    print(f"      Train ROC AUC: {train_auc:.4f} | Val ROC AUC: {val_auc:.4f} | Test ROC AUC: {test_auc:.4f}")
    print(f"      Train PR AUC:  {train_pr:.4f} | Val PR AUC:  {val_pr:.4f} | Test PR AUC:  {test_pr:.4f}")

# ============================================================================
# 3. Baseline Model 2: XGBoost with Class Balancing
# ============================================================================
print("\n" + "="*80)
print("3. BASELINE MODEL 2: XGBoost with Class Balancing")
print("="*80)

for feature_set_name in ['top_20', 'top_30', 'top_40', 'all_features']:
    print(f"\n   Testing feature set: {feature_set_name}")
    
    # Select features
    if feature_set_name == 'all_features':
        features = [col for col in train_df.columns if col not in ['CTRCD', 'time']]
    else:
        features = feature_sets[feature_set_name]
    
    X_train = train_df[features]
    y_train = train_df['CTRCD']
    X_val = val_df[features]
    y_val = val_df['CTRCD']
    X_test = test_df[features]
    y_test = test_df['CTRCD']
    
    print(f"      Features: {len(features)}")
    
    # Train model with scale_pos_weight
    model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )
    
    model.fit(X_train, y_train)
    
    # Predictions
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    
    # Evaluate
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    train_pr = average_precision_score(y_train, y_train_pred)
    val_pr = average_precision_score(y_val, y_val_pred)
    test_pr = average_precision_score(y_test, y_test_pred)
    
    results[f'baseline_balanced_{feature_set_name}'] = {
        'feature_set': feature_set_name,
        'n_features': len(features),
        'train_roc_auc': train_auc,
        'val_roc_auc': val_auc,
        'test_roc_auc': test_auc,
        'train_pr_auc': train_pr,
        'val_pr_auc': val_pr,
        'test_pr_auc': test_pr,
        'scale_pos_weight': scale_pos_weight,
        'model_type': 'balanced'
    }
    
    print(f"      Train ROC AUC: {train_auc:.4f} | Val ROC AUC: {val_auc:.4f} | Test ROC AUC: {test_auc:.4f}")
    print(f"      Train PR AUC:  {train_pr:.4f} | Val PR AUC:  {val_pr:.4f} | Test PR AUC:  {test_pr:.4f}")

# ============================================================================
# 4. Select Best Baseline Configuration
# ============================================================================
print("\n" + "="*80)
print("4. BASELINE MODEL COMPARISON")
print("="*80)

# Create comparison DataFrame
results_df = pd.DataFrame(results).T
results_df = results_df.sort_values('val_roc_auc', ascending=False)

print("\n   All Baseline Results (sorted by validation ROC AUC):")
print(results_df[['feature_set', 'n_features', 'val_roc_auc', 'test_roc_auc', 'val_pr_auc', 'test_pr_auc', 'model_type']].to_string())

# Save results
results_df.to_csv(TABLES_DIR / 'baseline_model_comparison.csv', index=True)
print(f"\n   ✓ Baseline comparison saved to: {TABLES_DIR / 'baseline_model_comparison.csv'}")

# Select best configuration
best_config = results_df.iloc[0].name
best_feature_set = results_df.iloc[0]['feature_set']
best_model_type = results_df.iloc[0]['model_type']

print(f"\n   Best baseline configuration: {best_config}")
print(f"      Feature set: {best_feature_set}")
print(f"      Model type: {best_model_type}")
print(f"      Validation ROC AUC: {results_df.iloc[0]['val_roc_auc']:.4f}")
print(f"      Test ROC AUC: {results_df.iloc[0]['test_roc_auc']:.4f}")

# ============================================================================
# 5. Train and Save Best Baseline Model
# ============================================================================
print("\n" + "="*80)
print("5. TRAINING AND SAVING BEST BASELINE MODEL")
print("="*80)

# Get features
if best_feature_set == 'all_features':
    best_features = [col for col in train_df.columns if col not in ['CTRCD', 'time']]
else:
    best_features = feature_sets[best_feature_set]

X_train = train_df[best_features]
y_train = train_df['CTRCD']
X_val = val_df[best_features]
y_val = val_df['CTRCD']
X_test = test_df[best_features]
y_test = test_df['CTRCD']

# Train best model
if best_model_type == 'balanced':
    best_model = xgb.XGBClassifier(
        objective='binary:logistic',
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        eval_metric='auc'
    )
else:
    best_model = xgb.XGBClassifier(
        objective='binary:logistic',
        random_state=42,
        eval_metric='auc'
    )

best_model.fit(X_train, y_train)

# Predictions
y_train_pred = best_model.predict_proba(X_train)[:, 1]
y_val_pred = best_model.predict_proba(X_val)[:, 1]
y_test_pred = best_model.predict_proba(X_test)[:, 1]

y_train_pred_class = best_model.predict(X_train)
y_val_pred_class = best_model.predict(X_val)
y_test_pred_class = best_model.predict(X_test)

# Save model
best_model.save_model(MODELS_DIR / 'baseline_model.json')
print(f"\n   ✓ Best baseline model saved to: {MODELS_DIR / 'baseline_model.json'}")

# Save feature list
with open(MODELS_DIR / 'baseline_features.json', 'w') as f:
    json.dump(best_features, f, indent=2)
print(f"   ✓ Feature list saved to: {MODELS_DIR / 'baseline_features.json'}")

# ============================================================================
# 6. Detailed Evaluation
# ============================================================================
print("\n" + "="*80)
print("6. DETAILED BASELINE MODEL EVALUATION")
print("="*80)

# ROC AUC
train_auc = roc_auc_score(y_train, y_train_pred)
val_auc = roc_auc_score(y_val, y_val_pred)
test_auc = roc_auc_score(y_test, y_test_pred)

# PR AUC
train_pr = average_precision_score(y_train, y_train_pred)
val_pr = average_precision_score(y_val, y_val_pred)
test_pr = average_precision_score(y_test, y_test_pred)

# F1 Score
train_f1 = f1_score(y_train, y_train_pred_class)
val_f1 = f1_score(y_val, y_val_pred_class)
test_f1 = f1_score(y_test, y_test_pred_class)

# Accuracy
train_acc = accuracy_score(y_train, y_train_pred_class)
val_acc = accuracy_score(y_val, y_val_pred_class)
test_acc = accuracy_score(y_test, y_test_pred_class)

print("\n   Performance Metrics:")
print(f"      {'Metric':<20} {'Train':<12} {'Validation':<12} {'Test':<12}")
print(f"      {'-'*56}")
print(f"      {'ROC AUC':<20} {train_auc:<12.4f} {val_auc:<12.4f} {test_auc:<12.4f}")
print(f"      {'PR AUC':<20} {train_pr:<12.4f} {val_pr:<12.4f} {test_pr:<12.4f}")
print(f"      {'F1 Score':<20} {train_f1:<12.4f} {val_f1:<12.4f} {test_f1:<12.4f}")
print(f"      {'Accuracy':<20} {train_acc:<12.4f} {val_acc:<12.4f} {test_acc:<12.4f}")

# Classification report
print("\n   Validation Set Classification Report:")
print(classification_report(y_val, y_val_pred_class, target_names=['No CTRCD', 'CTRCD']))

print("\n   Test Set Classification Report:")
print(classification_report(y_test, y_test_pred_class, target_names=['No CTRCD', 'CTRCD']))

# Confusion matrices
print("\n   Confusion Matrices:")
print("\n   Validation Set:")
cm_val = confusion_matrix(y_val, y_val_pred_class)
print(f"      {cm_val}")
print(f"      TN: {cm_val[0,0]}, FP: {cm_val[0,1]}, FN: {cm_val[1,0]}, TP: {cm_val[1,1]}")

print("\n   Test Set:")
cm_test = confusion_matrix(y_test, y_test_pred_class)
print(f"      {cm_test}")
print(f"      TN: {cm_test[0,0]}, FP: {cm_test[0,1]}, FN: {cm_test[1,0]}, TP: {cm_test[1,1]}")

# ============================================================================
# 7. Visualizations
# ============================================================================
print("\n" + "="*80)
print("7. CREATING BASELINE VISUALIZATIONS")
print("="*80)

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# 7.1 ROC Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (X, y, y_pred, split_name) in enumerate([
    (X_train, y_train, y_train_pred, 'Training'),
    (X_val, y_val, y_val_pred, 'Validation'),
    (X_test, y_test, y_test_pred, 'Test')
]):
    fpr, tpr, _ = roc_curve(y, y_pred)
    auc_score = roc_auc_score(y, y_pred)
    
    axes[idx].plot(fpr, tpr, linewidth=2, label=f'ROC (AUC = {auc_score:.4f})')
    axes[idx].plot([0, 1], [0, 1], 'k--', linewidth=1, label='Random')
    axes[idx].set_xlabel('False Positive Rate', fontsize=11)
    axes[idx].set_ylabel('True Positive Rate', fontsize=11)
    axes[idx].set_title(f'{split_name} Set ROC Curve', fontsize=13, fontweight='bold')
    axes[idx].legend(loc='lower right', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '11_baseline_roc_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ ROC curves saved to: {RESULTS_DIR / '11_baseline_roc_curves.png'}")

# 7.2 Precision-Recall Curves
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (X, y, y_pred, split_name) in enumerate([
    (X_train, y_train, y_train_pred, 'Training'),
    (X_val, y_val, y_val_pred, 'Validation'),
    (X_test, y_test, y_test_pred, 'Test')
]):
    precision, recall, _ = precision_recall_curve(y, y_pred)
    pr_auc = average_precision_score(y, y_pred)
    baseline_precision = y.sum() / len(y)
    
    axes[idx].plot(recall, precision, linewidth=2, label=f'PR (AUC = {pr_auc:.4f})')
    axes[idx].axhline(y=baseline_precision, color='k', linestyle='--', linewidth=1, 
                     label=f'Baseline ({baseline_precision:.4f})')
    axes[idx].set_xlabel('Recall', fontsize=11)
    axes[idx].set_ylabel('Precision', fontsize=11)
    axes[idx].set_title(f'{split_name} Set Precision-Recall Curve', fontsize=13, fontweight='bold')
    axes[idx].legend(loc='upper right', fontsize=10)
    axes[idx].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(RESULTS_DIR / '12_baseline_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ PR curves saved to: {RESULTS_DIR / '12_baseline_pr_curves.png'}")

# 7.3 Confusion Matrices
fig, axes = plt.subplots(1, 3, figsize=(18, 5))

for idx, (y_true, y_pred_class, split_name) in enumerate([
    (y_train, y_train_pred_class, 'Training'),
    (y_val, y_val_pred_class, 'Validation'),
    (y_test, y_test_pred_class, 'Test')
]):
    cm = confusion_matrix(y_true, y_pred_class)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[idx], 
                cbar_kws={'label': 'Count'})
    axes[idx].set_xlabel('Predicted Label', fontsize=11)
    axes[idx].set_ylabel('True Label', fontsize=11)
    axes[idx].set_title(f'{split_name} Set Confusion Matrix', fontsize=13, fontweight='bold')
    axes[idx].set_xticklabels(['No CTRCD', 'CTRCD'])
    axes[idx].set_yticklabels(['No CTRCD', 'CTRCD'])

plt.tight_layout()
plt.savefig(RESULTS_DIR / '13_baseline_confusion_matrices.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Confusion matrices saved to: {RESULTS_DIR / '13_baseline_confusion_matrices.png'}")

# 7.4 Feature Importance (Top 20)
fig, ax = plt.subplots(figsize=(10, 8))

importance_df = pd.DataFrame({
    'feature': best_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False).head(20)

ax.barh(range(len(importance_df)), importance_df['importance'])
ax.set_yticks(range(len(importance_df)))
ax.set_yticklabels(importance_df['feature'])
ax.invert_yaxis()
ax.set_xlabel('Importance', fontsize=11)
ax.set_title('Top 20 Feature Importances (Baseline Model)', fontsize=13, fontweight='bold')
ax.grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig(RESULTS_DIR / '14_baseline_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Feature importance plot saved to: {RESULTS_DIR / '14_baseline_feature_importance.png'}")

# Save all feature importances
importance_df_full = pd.DataFrame({
    'feature': best_features,
    'importance': best_model.feature_importances_
}).sort_values('importance', ascending=False)

importance_df_full.to_csv(TABLES_DIR / 'baseline_feature_importance.csv', index=False)
print(f"   ✓ Feature importance table saved to: {TABLES_DIR / 'baseline_feature_importance.csv'}")

# ============================================================================
# 8. Save Benchmark Metrics
# ============================================================================
print("\n" + "="*80)
print("8. SAVING BASELINE BENCHMARK METRICS")
print("="*80)

benchmark_metrics = {
    'model_type': best_model_type,
    'feature_set': best_feature_set,
    'n_features': len(best_features),
    'scale_pos_weight': float(scale_pos_weight) if best_model_type == 'balanced' else None,
    'metrics': {
        'train': {
            'roc_auc': float(train_auc),
            'pr_auc': float(train_pr),
            'f1_score': float(train_f1),
            'accuracy': float(train_acc)
        },
        'validation': {
            'roc_auc': float(val_auc),
            'pr_auc': float(val_pr),
            'f1_score': float(val_f1),
            'accuracy': float(val_acc)
        },
        'test': {
            'roc_auc': float(test_auc),
            'pr_auc': float(test_pr),
            'f1_score': float(test_f1),
            'accuracy': float(test_acc)
        }
    },
    'confusion_matrix': {
        'validation': {
            'TN': int(cm_val[0,0]),
            'FP': int(cm_val[0,1]),
            'FN': int(cm_val[1,0]),
            'TP': int(cm_val[1,1])
        },
        'test': {
            'TN': int(cm_test[0,0]),
            'FP': int(cm_test[0,1]),
            'FN': int(cm_test[1,0]),
            'TP': int(cm_test[1,1])
        }
    }
}

with open(MODELS_DIR / 'baseline_metrics.json', 'w') as f:
    json.dump(benchmark_metrics, f, indent=2)

print(f"   ✓ Baseline metrics saved to: {MODELS_DIR / 'baseline_metrics.json'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 5 COMPLETE: BASELINE MODEL DEVELOPMENT")
print("="*80)

print(f"\n✓ Baseline model established and saved")
print(f"✓ Feature set: {best_feature_set} ({len(best_features)} features)")
print(f"✓ Model type: {best_model_type}")
print(f"\n✓ Performance benchmarks:")
print(f"   - Validation ROC AUC: {val_auc:.4f}")
print(f"   - Test ROC AUC: {test_auc:.4f}")
print(f"   - Validation PR AUC: {val_pr:.4f}")
print(f"   - Test PR AUC: {test_pr:.4f}")
print(f"\n✓ Files saved:")
print(f"   - Model: {MODELS_DIR / 'baseline_model.json'}")
print(f"   - Features: {MODELS_DIR / 'baseline_features.json'}")
print(f"   - Metrics: {MODELS_DIR / 'baseline_metrics.json'}")
print(f"   - Feature importance: {TABLES_DIR / 'baseline_feature_importance.csv'}")
print(f"   - Model comparison: {TABLES_DIR / 'baseline_model_comparison.csv'}")
print(f"   - Visualizations: 4 figures saved to {RESULTS_DIR}")

print("\n" + "="*80)
print("Ready for Phase 6: Model Optimization")
print("="*80)

