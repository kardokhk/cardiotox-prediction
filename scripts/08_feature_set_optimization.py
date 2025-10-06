"""
Phase 9: Feature Set Optimization
Test optimized hyperparameters across different feature sets from Phase 4

Goals:
1. Load optimized hyperparameters from Phase 7
2. Test across all feature sets: top_20, top_30, top_40, top_50, rfe_selected, significant, all_features
3. Compare performance with stratified cross-validation
4. Identify optimal feature set for the optimized model
5. Evaluate best configuration on test set

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
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.metrics import (
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report,
    roc_curve, precision_recall_curve
)
import warnings
warnings.filterwarnings('ignore')

# Set style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Paths
BASE_DIR = Path(__file__).parent.parent
ENGINEERED_DIR = BASE_DIR / 'data' / 'engineered'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results' / 'figures'
TABLES_DIR = BASE_DIR / 'results' / 'tables'

print("="*80)
print("PHASE 9: FEATURE SET OPTIMIZATION")
print("="*80)

# Load data
print("\n1. Loading data and configurations...")
train_df = pd.read_csv(ENGINEERED_DIR / 'train_engineered.csv')
val_df = pd.read_csv(ENGINEERED_DIR / 'val_engineered.csv')
test_df = pd.read_csv(ENGINEERED_DIR / 'test_engineered.csv')

# Load feature sets from Phase 4
with open(ENGINEERED_DIR / 'feature_sets.json', 'r') as f:
    feature_sets = json.load(f)

# Load optimized hyperparameters from Phase 7
with open(MODELS_DIR / 'random_search_metrics.json', 'r') as f:
    best_config = json.load(f)
    optimized_params = best_config['best_params']

print(f"   Training samples: {len(train_df)}")
print(f"   Validation samples: {len(val_df)}")
print(f"   Test samples: {len(test_df)}")
print(f"\n   Available feature sets:")
for name, features in feature_sets.items():
    print(f"      - {name}: {len(features)} features")

print(f"\n   Optimized hyperparameters loaded from Phase 7:")
for param, value in sorted(optimized_params.items()):
    print(f"      {param}: {value}")

# ============================================================================
# 2. Test Each Feature Set with Optimized Hyperparameters
# ============================================================================
print("\n" + "="*80)
print("2. TESTING FEATURE SETS WITH OPTIMIZED HYPERPARAMETERS")
print("="*80)

# Stratified k-fold CV
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# Define scoring metrics
scoring = {
    'roc_auc': 'roc_auc',
    'pr_auc': 'average_precision',
    'accuracy': 'accuracy',
    'balanced_accuracy': 'balanced_accuracy',
    'f1': 'f1',
    'precision': 'precision',
    'recall': 'recall'
}

# Store results
results_list = []

# Test each feature set
for feature_set_name in ['top_20', 'top_30', 'top_40', 'top_50', 
                          'rfe_selected', 'high_importance', 'significant', 'all_features']:
    
    print(f"\n   Testing: {feature_set_name}")
    print("   " + "-"*76)
    
    # Get features for this set
    features = feature_sets[feature_set_name]
    
    # Skip empty feature sets
    if len(features) == 0:
        print(f"      ⚠ Skipping: No features in this set")
        continue
    
    # Prepare data
    X_train = train_df[features]
    y_train = train_df['CTRCD']
    X_val = val_df[features]
    y_val = val_df['CTRCD']
    X_test = test_df[features]
    y_test = test_df['CTRCD']
    
    print(f"      Features: {len(features)}")
    
    # Create model with optimized hyperparameters
    model = xgb.XGBClassifier(
        **optimized_params,
        objective='binary:logistic',
        eval_metric='auc',
        random_state=42,
        n_jobs=-1
    )
    
    # Cross-validation on training set
    print(f"      Running 5-fold CV on training set...")
    cv_results = cross_validate(
        model, X_train, y_train,
        cv=cv,
        scoring=scoring,
        return_train_score=True,
        n_jobs=-1
    )
    
    cv_train_auc = cv_results['train_roc_auc'].mean()
    cv_train_auc_std = cv_results['train_roc_auc'].std()
    cv_test_auc = cv_results['test_roc_auc'].mean()
    cv_test_auc_std = cv_results['test_roc_auc'].std()
    cv_pr_auc = cv_results['test_pr_auc'].mean()
    cv_pr_auc_std = cv_results['test_pr_auc'].std()
    
    # Train on full training set
    model.fit(X_train, y_train)
    
    # Evaluate on validation and test sets
    y_train_pred = model.predict_proba(X_train)[:, 1]
    y_val_pred = model.predict_proba(X_val)[:, 1]
    y_test_pred = model.predict_proba(X_test)[:, 1]
    
    train_auc = roc_auc_score(y_train, y_train_pred)
    val_auc = roc_auc_score(y_val, y_val_pred)
    test_auc = roc_auc_score(y_test, y_test_pred)
    
    train_pr = average_precision_score(y_train, y_train_pred)
    val_pr = average_precision_score(y_val, y_val_pred)
    test_pr = average_precision_score(y_test, y_test_pred)
    
    # Calculate overfitting metric
    overfitting_gap = train_auc - cv_test_auc
    
    print(f"      CV ROC AUC: {cv_test_auc:.4f} ± {cv_test_auc_std:.4f}")
    print(f"      Train ROC AUC: {train_auc:.4f}")
    print(f"      Val ROC AUC: {val_auc:.4f}")
    print(f"      Test ROC AUC: {test_auc:.4f}")
    print(f"      Overfitting gap: {overfitting_gap:.4f}")
    
    # Store results
    results_list.append({
        'feature_set': feature_set_name,
        'n_features': len(features),
        'cv_train_auc': cv_train_auc,
        'cv_train_auc_std': cv_train_auc_std,
        'cv_test_auc': cv_test_auc,
        'cv_test_auc_std': cv_test_auc_std,
        'cv_pr_auc': cv_pr_auc,
        'cv_pr_auc_std': cv_pr_auc_std,
        'train_auc': train_auc,
        'val_auc': val_auc,
        'test_auc': test_auc,
        'train_pr': train_pr,
        'val_pr': val_pr,
        'test_pr': test_pr,
        'overfitting_gap': overfitting_gap
    })

# ============================================================================
# 3. Compare Results
# ============================================================================
print("\n" + "="*80)
print("3. COMPARING FEATURE SET PERFORMANCE")
print("="*80)

# Create results DataFrame
results_df = pd.DataFrame(results_list)

# Sort by CV test AUC (most reliable metric)
results_df = results_df.sort_values('cv_test_auc', ascending=False)

print("\n   Feature Set Comparison (sorted by CV ROC AUC):")
print("   " + "-"*76)
print(results_df[['feature_set', 'n_features', 'cv_test_auc', 'val_auc', 
                   'test_auc', 'overfitting_gap']].to_string(index=False))

# Save results
results_df.to_csv(TABLES_DIR / '19_feature_set_optimization_results.csv', index=False)
print(f"\n   ✓ Results saved to: {TABLES_DIR / '19_feature_set_optimization_results.csv'}")

# Identify best feature set
best_idx = results_df.iloc[0]
print(f"\n   🏆 Best Feature Set: {best_idx['feature_set']}")
print(f"      Features: {best_idx['n_features']}")
print(f"      CV ROC AUC: {best_idx['cv_test_auc']:.4f} ± {best_idx['cv_test_auc_std']:.4f}")
print(f"      Validation ROC AUC: {best_idx['val_auc']:.4f}")
print(f"      Test ROC AUC: {best_idx['test_auc']:.4f}")
print(f"      Test PR AUC: {best_idx['test_pr']:.4f}")
print(f"      Overfitting gap: {best_idx['overfitting_gap']:.4f}")

# Compare with Phase 7 baseline (all_features)
baseline_idx = results_df[results_df['feature_set'] == 'all_features'].iloc[0]
improvement = ((best_idx['test_auc'] - baseline_idx['test_auc']) / baseline_idx['test_auc']) * 100

print(f"\n   📊 Comparison with Phase 7 (all_features):")
print(f"      Phase 7 (88 features): Test AUC = {baseline_idx['test_auc']:.4f}")
print(f"      Phase 9 (Best set): Test AUC = {best_idx['test_auc']:.4f}")
if improvement > 0:
    print(f"      Improvement: +{improvement:.2f}%")
    print(f"      Feature reduction: {baseline_idx['n_features']} → {best_idx['n_features']} ({((baseline_idx['n_features'] - best_idx['n_features']) / baseline_idx['n_features'] * 100):.1f}% fewer features)")
elif improvement < 0:
    print(f"      Change: {improvement:.2f}%")
else:
    print(f"      No significant change")

# ============================================================================
# 4. Visualizations
# ============================================================================
print("\n" + "="*80)
print("4. CREATING VISUALIZATIONS")
print("="*80)

# 4.1 CV ROC AUC by Feature Set
fig, ax = plt.subplots(figsize=(14, 8))
results_sorted = results_df.sort_values('cv_test_auc')
colors = ['#2ecc71' if x == best_idx['feature_set'] else '#3498db' 
          for x in results_sorted['feature_set']]

bars = ax.barh(results_sorted['feature_set'], results_sorted['cv_test_auc'], 
               color=colors, alpha=0.7, edgecolor='black')
ax.errorbar(results_sorted['cv_test_auc'], range(len(results_sorted)), 
            xerr=results_sorted['cv_test_auc_std'], fmt='none', 
            ecolor='black', capsize=5, alpha=0.6)

# Add feature count labels
for i, (idx, row) in enumerate(results_sorted.iterrows()):
    ax.text(row['cv_test_auc'] + 0.01, i, f"n={row['n_features']}", 
            va='center', fontsize=9)

ax.set_xlabel('CV ROC AUC (5-fold)', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature Set', fontsize=12, fontweight='bold')
ax.set_title('Feature Set Performance with Optimized Hyperparameters\n(Error bars show ±1 SD)', 
             fontsize=14, fontweight='bold')
ax.axvline(x=baseline_idx['cv_test_auc'], color='red', linestyle='--', 
           linewidth=2, alpha=0.5, label=f"Phase 7 baseline ({baseline_idx['cv_test_auc']:.4f})")
ax.legend(fontsize=10)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '23_feature_set_cv_performance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 23_feature_set_cv_performance.png")

# 4.2 Test AUC vs Number of Features
fig, ax = plt.subplots(figsize=(12, 7))
ax.scatter(results_df['n_features'], results_df['test_auc'], 
           s=200, alpha=0.6, c=results_df['cv_test_auc'], 
           cmap='RdYlGn', edgecolor='black', linewidth=1.5)

# Annotate points
for idx, row in results_df.iterrows():
    ax.annotate(row['feature_set'], 
                (row['n_features'], row['test_auc']),
                xytext=(5, 5), textcoords='offset points',
                fontsize=9, alpha=0.8)

# Highlight best
best_row = results_df.iloc[0]
ax.scatter(best_row['n_features'], best_row['test_auc'], 
           s=400, alpha=0.3, color='gold', edgecolor='orange', linewidth=3,
           label='Best feature set')

ax.set_xlabel('Number of Features', fontsize=12, fontweight='bold')
ax.set_ylabel('Test ROC AUC', fontsize=12, fontweight='bold')
ax.set_title('Test Performance vs Feature Set Size\n(Color indicates CV AUC)', 
             fontsize=14, fontweight='bold')
ax.grid(True, alpha=0.3)
ax.legend(fontsize=10)
plt.colorbar(ax.collections[0], ax=ax, label='CV ROC AUC')
plt.tight_layout()
plt.savefig(RESULTS_DIR / '24_test_auc_vs_features.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 24_test_auc_vs_features.png")

# 4.3 Overfitting Analysis
fig, ax = plt.subplots(figsize=(12, 7))
results_sorted = results_df.sort_values('overfitting_gap')
colors = ['#e74c3c' if x > 0.1 else '#f39c12' if x > 0.05 else '#2ecc71' 
          for x in results_sorted['overfitting_gap']]

bars = ax.barh(results_sorted['feature_set'], results_sorted['overfitting_gap'], 
               color=colors, alpha=0.7, edgecolor='black')
ax.set_xlabel('Overfitting Gap (Train AUC - CV Test AUC)', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature Set', fontsize=12, fontweight='bold')
ax.set_title('Overfitting Analysis by Feature Set\n(Green: Low, Orange: Moderate, Red: High)', 
             fontsize=14, fontweight='bold')
ax.axvline(x=0.05, color='orange', linestyle='--', linewidth=1, alpha=0.5)
ax.axvline(x=0.1, color='red', linestyle='--', linewidth=1, alpha=0.5)
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '25_overfitting_analysis.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 25_overfitting_analysis.png")

# 4.4 Comprehensive Comparison Heatmap
fig, ax = plt.subplots(figsize=(10, 8))
heatmap_data = results_df[['feature_set', 'cv_test_auc', 'val_auc', 'test_auc', 
                            'cv_pr_auc', 'val_pr', 'test_pr']].set_index('feature_set')
heatmap_data.columns = ['CV ROC AUC', 'Val ROC AUC', 'Test ROC AUC', 
                        'CV PR AUC', 'Val PR AUC', 'Test PR AUC']
sns.heatmap(heatmap_data, annot=True, fmt='.4f', cmap='RdYlGn', 
            center=0.70, vmin=0.65, vmax=0.80, ax=ax, 
            cbar_kws={'label': 'AUC Score'})
ax.set_xlabel('Metric', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature Set', fontsize=12, fontweight='bold')
ax.set_title('Comprehensive Performance Heatmap', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig(RESULTS_DIR / '26_performance_heatmap.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Saved: 26_performance_heatmap.png")

# ============================================================================
# 5. Train and Evaluate Final Model with Best Feature Set
# ============================================================================
print("\n" + "="*80)
print("5. TRAINING FINAL MODEL WITH BEST FEATURE SET")
print("="*80)

best_feature_set_name = best_idx['feature_set']
best_features = feature_sets[best_feature_set_name]

print(f"\n   Selected feature set: {best_feature_set_name}")
print(f"   Number of features: {len(best_features)}")

# Prepare data
X_train = train_df[best_features]
y_train = train_df['CTRCD']
X_val = val_df[best_features]
y_val = val_df['CTRCD']
X_test = test_df[best_features]
y_test = test_df['CTRCD']

# Train final model
final_model = xgb.XGBClassifier(
    **optimized_params,
    objective='binary:logistic',
    eval_metric='auc',
    random_state=42,
    n_jobs=-1
)

final_model.fit(X_train, y_train)
print(f"   ✓ Model trained successfully")

# Predictions
y_train_proba = final_model.predict_proba(X_train)[:, 1]
y_val_proba = final_model.predict_proba(X_val)[:, 1]
y_test_proba = final_model.predict_proba(X_test)[:, 1]

y_train_pred = (y_train_proba >= 0.5).astype(int)
y_val_pred = (y_val_proba >= 0.5).astype(int)
y_test_pred = (y_test_proba >= 0.5).astype(int)

# Comprehensive metrics
print("\n   Final Model Performance:")
print("   " + "-"*76)
print(f"   {'Metric':<20} {'Train':>15} {'Validation':>15} {'Test':>15}")
print("   " + "-"*76)

metrics = [
    ('ROC AUC', roc_auc_score),
    ('PR AUC', average_precision_score)
]

for metric_name, metric_func in metrics:
    train_score = metric_func(y_train, y_train_proba)
    val_score = metric_func(y_val, y_val_proba)
    test_score = metric_func(y_test, y_test_proba)
    print(f"   {metric_name:<20} {train_score:>15.4f} {val_score:>15.4f} {test_score:>15.4f}")

print("   " + "-"*76)

# Confusion matrices
print("\n   Confusion Matrices:")
for name, y_true, y_pred in [('Train', y_train, y_train_pred), 
                               ('Validation', y_val, y_val_pred),
                               ('Test', y_test, y_test_pred)]:
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    print(f"\n   {name}:")
    print(f"      TN: {tn:>3}  FP: {fp:>3}")
    print(f"      FN: {fn:>3}  TP: {tp:>3}")
    if (tp + fn) > 0:
        sensitivity = tp / (tp + fn)
        print(f"      Sensitivity: {sensitivity:.4f}")
    if (tn + fp) > 0:
        specificity = tn / (tn + fp)
        print(f"      Specificity: {specificity:.4f}")

# ============================================================================
# 6. Feature Importance for Best Model
# ============================================================================
print("\n" + "="*80)
print("6. FEATURE IMPORTANCE ANALYSIS")
print("="*80)

# Get feature importance
importance_df = pd.DataFrame({
    'feature': best_features,
    'importance': final_model.feature_importances_
}).sort_values('importance', ascending=False)

print(f"\n   Top 15 Most Important Features:")
print(importance_df.head(15).to_string(index=False))

# Save full importance
importance_df.to_csv(TABLES_DIR / '20_optimized_feature_importance.csv', index=False)
print(f"\n   ✓ Full feature importance saved to: {TABLES_DIR / '20_optimized_feature_importance.csv'}")

# Visualize top 20 features
n_features_to_plot = min(20, len(best_features))
fig, ax = plt.subplots(figsize=(10, max(8, n_features_to_plot * 0.4)))
top_features = importance_df.head(n_features_to_plot).sort_values('importance')
ax.barh(top_features['feature'], top_features['importance'], 
        color='#3498db', alpha=0.7, edgecolor='black')
ax.set_xlabel('Importance', fontsize=12, fontweight='bold')
ax.set_ylabel('Feature', fontsize=12, fontweight='bold')
ax.set_title(f'Top {n_features_to_plot} Feature Importances\n({best_feature_set_name})', 
             fontsize=14, fontweight='bold')
ax.grid(axis='x', alpha=0.3)
plt.tight_layout()
plt.savefig(RESULTS_DIR / '27_optimized_feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"   ✓ Feature importance plot saved to: {RESULTS_DIR / '27_optimized_feature_importance.png'}")

# ============================================================================
# 7. Save Final Optimized Model
# ============================================================================
print("\n" + "="*80)
print("7. SAVING FINAL OPTIMIZED MODEL")
print("="*80)

# Save model
final_model.save_model(MODELS_DIR / 'optimized_model_final.json')
print(f"   ✓ Model saved to: {MODELS_DIR / 'optimized_model_final.json'}")

# Save feature list
with open(MODELS_DIR / 'optimized_features_final.json', 'w') as f:
    json.dump(best_features, f, indent=2)
print(f"   ✓ Features saved to: {MODELS_DIR / 'optimized_features_final.json'}")

# Save comprehensive results
final_results = {
    'phase': 'Phase 9: Feature Set Optimization',
    'date': '2025-10-06',
    'best_feature_set': best_feature_set_name,
    'n_features': len(best_features),
    'hyperparameters': optimized_params,
    'performance': {
        'cv_roc_auc': float(best_idx['cv_test_auc']),
        'cv_roc_auc_std': float(best_idx['cv_test_auc_std']),
        'cv_pr_auc': float(best_idx['cv_pr_auc']),
        'train_roc_auc': float(roc_auc_score(y_train, y_train_proba)),
        'val_roc_auc': float(roc_auc_score(y_val, y_val_proba)),
        'test_roc_auc': float(roc_auc_score(y_test, y_test_proba)),
        'train_pr_auc': float(average_precision_score(y_train, y_train_proba)),
        'val_pr_auc': float(average_precision_score(y_val, y_val_proba)),
        'test_pr_auc': float(average_precision_score(y_test, y_test_proba)),
        'overfitting_gap': float(best_idx['overfitting_gap'])
    },
    'comparison_with_phase7': {
        'phase7_features': int(baseline_idx['n_features']),
        'phase9_features': len(best_features),
        'feature_reduction_pct': float(((baseline_idx['n_features'] - len(best_features)) / baseline_idx['n_features'] * 100)),
        'phase7_test_auc': float(baseline_idx['test_auc']),
        'phase9_test_auc': float(best_idx['test_auc']),
        'improvement_pct': float(improvement)
    },
    'all_feature_sets_tested': results_df.to_dict('records')
}

with open(MODELS_DIR / 'optimized_results_phase9.json', 'w') as f:
    json.dump(final_results, f, indent=2)
print(f"   ✓ Results saved to: {MODELS_DIR / 'optimized_results_phase9.json'}")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 9 COMPLETE: FEATURE SET OPTIMIZATION")
print("="*80)

print(f"\n✓ Tested {len(feature_sets)} different feature sets with optimized hyperparameters")
print(f"✓ Best feature set: {best_feature_set_name} ({len(best_features)} features)")
print(f"\n📊 Performance Summary:")
print(f"   CV ROC AUC: {best_idx['cv_test_auc']:.4f} ± {best_idx['cv_test_auc_std']:.4f}")
print(f"   Validation ROC AUC: {best_idx['val_auc']:.4f}")
print(f"   Test ROC AUC: {best_idx['test_auc']:.4f}")
print(f"   Test PR AUC: {best_idx['test_pr']:.4f}")

print(f"\n📈 Comparison with Phase 7:")
print(f"   Phase 7: {baseline_idx['n_features']} features, Test AUC = {baseline_idx['test_auc']:.4f}")
print(f"   Phase 9: {len(best_features)} features, Test AUC = {best_idx['test_auc']:.4f}")
if improvement > 0:
    print(f"   ✓ Improvement: +{improvement:.2f}%")
    print(f"   ✓ Feature reduction: {((baseline_idx['n_features'] - len(best_features)) / baseline_idx['n_features'] * 100):.1f}%")
elif improvement < 0:
    print(f"   ⚠ Performance change: {improvement:.2f}%")
    if len(best_features) < baseline_idx['n_features']:
        print(f"   ✓ But achieved {((baseline_idx['n_features'] - len(best_features)) / baseline_idx['n_features'] * 100):.1f}% feature reduction")
        print(f"   ✓ Better model interpretability with minimal performance impact")

print(f"\n📁 Files saved:")
print(f"   - Model: {MODELS_DIR / 'optimized_model_final.json'}")
print(f"   - Features: {MODELS_DIR / 'optimized_features_final.json'}")
print(f"   - Results: {MODELS_DIR / 'optimized_results_phase9.json'}")
print(f"   - Comparison table: {TABLES_DIR / '19_feature_set_optimization_results.csv'}")
print(f"   - Feature importance: {TABLES_DIR / '20_optimized_feature_importance.csv'}")
print(f"   - Visualizations: 5 new figures saved to {RESULTS_DIR}")

print("\n" + "="*80)
print("✓ Feature set optimization complete!")
print("✓ Ready for Phase 9: Model Interpretation")
print("="*80)
