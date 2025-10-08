"""
Phase 10: Model Documentation
Comprehensive documentation and visualization for the best model (RFE-selected, 40 features)
- ROC curves with train/validation/test splits
- Precision-Recall curves with train/validation/test splits
- Confusion matrices for all splits
- Performance comparison tables
- Hyperparameter documentation
- Feature importance rankings

Author: Kardokh Kaka Bra
Date: October 2025
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import xgboost as xgb
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    roc_curve, auc, precision_recall_curve, average_precision_score,
    confusion_matrix, classification_report, RocCurveDisplay,
    PrecisionRecallDisplay
)
import warnings
warnings.filterwarnings('ignore')

# Set elegant academic style
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_context("paper", font_scale=1.4)
sns.set_palette("colorblind")

# Custom colors for consistency
COLORS = {
    'train': '#2E86AB',      # Blue
    'validation': '#A23B72',  # Purple
    'test': '#F18F01',       # Orange
    'positive': '#C73E1D',   # Red
    'negative': '#6A994E'    # Green
}

# Paths
BASE_DIR = Path(__file__).parent.parent
ENGINEERED_DIR = BASE_DIR / 'data' / 'engineered'
MODELS_DIR = BASE_DIR / 'models'
RESULTS_DIR = BASE_DIR / 'results'
FIGURES_DIR = RESULTS_DIR / 'figures'
TABLES_DIR = RESULTS_DIR / 'tables'
TABLES_DIR.mkdir(parents=True, exist_ok=True)

print("="*80)
print("PHASE 10: MODEL DOCUMENTATION")
print("Best Model: RFE-Selected Features (40 features)")
print("="*80)

# ============================================================================
# 1. Load Best Model and Data
# ============================================================================
print("\n1. Loading best model and data...")

# Load model
model = xgb.XGBClassifier()
model.load_model(MODELS_DIR / 'final_best_model.json')
print(f"   ✓ Model loaded: final_best_model.json")

# Load features
with open(MODELS_DIR / 'final_best_features.json', 'r') as f:
    features = json.load(f)
print(f"   ✓ Features loaded: {len(features)} features")

# Load model card
with open(MODELS_DIR / 'final_best_model_card.json', 'r') as f:
    model_card = json.load(f)
print(f"   ✓ Model card loaded")
print(f"      Test ROC AUC: {model_card['performance']['test']['roc_auc']:.4f}")

# Load datasets
train_df = pd.read_csv(ENGINEERED_DIR / 'train_engineered.csv')
val_df = pd.read_csv(ENGINEERED_DIR / 'val_engineered.csv')
test_df = pd.read_csv(ENGINEERED_DIR / 'test_engineered.csv')

X_train = train_df[features]
y_train = train_df['CTRCD']
X_val = val_df[features]
y_val = val_df['CTRCD']
X_test = test_df[features]
y_test = test_df['CTRCD']

print(f"\n   Dataset sizes:")
print(f"      Train: {len(X_train)} ({y_train.sum()} positive)")
print(f"      Validation: {len(X_val)} ({y_val.sum()} positive)")
print(f"      Test: {len(X_test)} ({y_test.sum()} positive)")

# ============================================================================
# 2. Generate Predictions
# ============================================================================
print("\n" + "="*80)
print("2. GENERATING PREDICTIONS")
print("="*80)

# Predict probabilities
y_train_proba = model.predict_proba(X_train)[:, 1]
y_val_proba = model.predict_proba(X_val)[:, 1]
y_test_proba = model.predict_proba(X_test)[:, 1]

# Binary predictions (using default threshold 0.5)
y_train_pred = model.predict(X_train)
y_val_pred = model.predict(X_val)
y_test_pred = model.predict(X_test)

print(f"   ✓ Predictions generated for all splits")
print(f"   Training prediction range: [{y_train_proba.min():.4f}, {y_train_proba.max():.4f}]")
print(f"   Validation prediction range: [{y_val_proba.min():.4f}, {y_val_proba.max():.4f}]")
print(f"   Test prediction range: [{y_test_proba.min():.4f}, {y_test_proba.max():.4f}]")

# ============================================================================
# 3. ROC Curves (Train/Val/Test)
# ============================================================================
print("\n" + "="*80)
print("3. ROC CURVES (TRAIN/VALIDATION/TEST)")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 8))

# Calculate ROC curves
splits = [
    ('Training', y_train, y_train_proba, COLORS['train']),
    ('Validation', y_val, y_val_proba, COLORS['validation']),
    ('Test', y_test, y_test_proba, COLORS['test'])
]

for split_name, y_true, y_proba, color in splits:
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = auc(fpr, tpr)
    ax.plot(fpr, tpr, color=color, lw=2.5, 
            label=f'{split_name} (AUC = {roc_auc:.4f})', alpha=0.85)
    print(f"   {split_name:12} ROC AUC: {roc_auc:.4f}")

# Diagonal reference line
ax.plot([0, 1], [0, 1], 'k--', lw=1.5, alpha=0.4, label='Random (AUC = 0.5000)')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('False Positive Rate (1 - Specificity)', fontsize=14, fontweight='bold')
ax.set_ylabel('True Positive Rate (Sensitivity)', fontsize=14, fontweight='bold')
ax.set_title('ROC Curves: Cardiotoxicity Prediction Model\n(RFE-Selected Features, n=40)', 
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='lower right', fontsize=12, frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add subtle background
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '28_final_roc_curves_documentation.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ ROC curves saved: 28_final_roc_curves_documentation.png")

# ============================================================================
# 4. Precision-Recall Curves (Train/Val/Test)
# ============================================================================
print("\n" + "="*80)
print("4. PRECISION-RECALL CURVES (TRAIN/VALIDATION/TEST)")
print("="*80)

fig, ax = plt.subplots(figsize=(10, 8))

for split_name, y_true, y_proba, color in splits:
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)
    ax.plot(recall, precision, color=color, lw=2.5,
            label=f'{split_name} (AP = {pr_auc:.4f})', alpha=0.85)
    print(f"   {split_name:12} PR AUC: {pr_auc:.4f}")

# Baseline (prevalence)
baseline_train = y_train.mean()
baseline_val = y_val.mean()
baseline_test = y_test.mean()
ax.axhline(y=baseline_test, color='k', linestyle='--', lw=1.5, alpha=0.4,
           label=f'Baseline (Prevalence = {baseline_test:.4f})')

ax.set_xlim([0.0, 1.0])
ax.set_ylim([0.0, 1.05])
ax.set_xlabel('Recall (Sensitivity)', fontsize=14, fontweight='bold')
ax.set_ylabel('Precision (PPV)', fontsize=14, fontweight='bold')
ax.set_title('Precision-Recall Curves: Cardiotoxicity Prediction Model\n(RFE-Selected Features, n=40)',
             fontsize=16, fontweight='bold', pad=20)
ax.legend(loc='upper right', fontsize=12, frameon=True, shadow=True)
ax.grid(True, alpha=0.3, linestyle='-', linewidth=0.5)

# Add subtle background
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '29_final_pr_curves_documentation.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ PR curves saved: 29_final_pr_curves_documentation.png")

# ============================================================================
# 5. Confusion Matrices (Train/Val/Test)
# ============================================================================
print("\n" + "="*80)
print("5. CONFUSION MATRICES (TRAIN/VALIDATION/TEST)")
print("="*80)

fig, axes = plt.subplots(1, 3, figsize=(18, 5))

splits_cm = [
    ('Training', y_train, y_train_pred, axes[0]),
    ('Validation', y_val, y_val_pred, axes[1]),
    ('Test', y_test, y_test_pred, axes[2])
]

for split_name, y_true, y_pred, ax in splits_cm:
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalize to percentages
    cm_pct = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] * 100
    
    # Create annotations with both counts and percentages
    annot = np.empty_like(cm).astype(str)
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            annot[i, j] = f'{cm[i, j]}\n({cm_pct[i, j]:.1f}%)'
    
    # Plot heatmap
    sns.heatmap(cm, annot=annot, fmt='', cmap='Blues', cbar=True,
                square=True, ax=ax, linewidths=2, linecolor='white',
                vmin=0, vmax=cm.max(), annot_kws={'fontsize': 12, 'fontweight': 'bold'})
    
    ax.set_xlabel('Predicted Label', fontsize=12, fontweight='bold')
    ax.set_ylabel('True Label', fontsize=12, fontweight='bold')
    ax.set_title(f'{split_name} Set\n(n={len(y_true)})', 
                 fontsize=14, fontweight='bold', pad=10)
    ax.set_xticklabels(['No CTRCD', 'CTRCD'], fontsize=11)
    ax.set_yticklabels(['No CTRCD', 'CTRCD'], fontsize=11, rotation=90, va='center')
    
    # Calculate metrics
    tn, fp, fn, tp = cm.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"\n   {split_name} Confusion Matrix:")
    print(f"      TN: {tn:3d}  FP: {fp:3d}")
    print(f"      FN: {fn:3d}  TP: {tp:3d}")
    print(f"      Sensitivity: {sensitivity:.4f}")
    print(f"      Specificity: {specificity:.4f}")

plt.suptitle('Confusion Matrices: Cardiotoxicity Prediction Model\n(RFE-Selected Features, n=40)',
             fontsize=16, fontweight='bold', y=1.08)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '30_final_confusion_matrices_documentation.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"\n   ✓ Confusion matrices saved: 30_final_confusion_matrices_documentation.png")

# ============================================================================
# 6. Performance Comparison Table
# ============================================================================
print("\n" + "="*80)
print("6. PERFORMANCE COMPARISON TABLE")
print("="*80)

# Calculate comprehensive metrics for all splits
from sklearn.metrics import (
    roc_auc_score, average_precision_score, accuracy_score,
    balanced_accuracy_score, f1_score, precision_score, recall_score
)

def calculate_metrics(y_true, y_pred, y_proba):
    """Calculate comprehensive metrics"""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    metrics = {
        'ROC AUC': roc_auc_score(y_true, y_proba),
        'PR AUC': average_precision_score(y_true, y_proba),
        'Accuracy': accuracy_score(y_true, y_pred),
        'Balanced Accuracy': balanced_accuracy_score(y_true, y_pred),
        'Sensitivity (Recall)': recall_score(y_true, y_pred, zero_division=0),
        'Specificity': tn / (tn + fp) if (tn + fp) > 0 else 0,
        'Precision (PPV)': precision_score(y_true, y_pred, zero_division=0),
        'F1 Score': f1_score(y_true, y_pred, zero_division=0),
        'True Negatives': tn,
        'False Positives': fp,
        'False Negatives': fn,
        'True Positives': tp
    }
    return metrics

# Calculate for all splits
train_metrics = calculate_metrics(y_train, y_train_pred, y_train_proba)
val_metrics = calculate_metrics(y_val, y_val_pred, y_val_proba)
test_metrics = calculate_metrics(y_test, y_test_pred, y_test_proba)

# Create comparison dataframe
metrics_df = pd.DataFrame({
    'Metric': list(train_metrics.keys()),
    'Training': list(train_metrics.values()),
    'Validation': list(val_metrics.values()),
    'Test': list(test_metrics.values())
})

# Format numeric columns
for col in ['Training', 'Validation', 'Test']:
    metrics_df[col] = metrics_df[col].apply(
        lambda x: f'{x:.4f}' if isinstance(x, float) and x < 10 else f'{int(x)}' if isinstance(x, float) else x
    )

# Save to CSV
metrics_df.to_csv(TABLES_DIR / 'performance_metrics_comparison.csv', index=False)

print("\n   Performance Metrics Comparison:")
print(metrics_df.to_string(index=False))
print(f"\n   ✓ Table saved: performance_metrics_comparison.csv")

# ============================================================================
# 7. Model Performance Heatmap
# ============================================================================
print("\n" + "="*80)
print("7. MODEL PERFORMANCE HEATMAP")
print("="*80)

# Select key metrics for heatmap
key_metrics = ['ROC AUC', 'PR AUC', 'Balanced Accuracy', 
               'Sensitivity (Recall)', 'Specificity', 'Precision (PPV)']

heatmap_data = metrics_df[metrics_df['Metric'].isin(key_metrics)].copy()
heatmap_data.set_index('Metric', inplace=True)

# Convert to numeric
for col in heatmap_data.columns:
    heatmap_data[col] = pd.to_numeric(heatmap_data[col])

fig, ax = plt.subplots(figsize=(10, 7))

sns.heatmap(heatmap_data.T, annot=True, fmt='.4f', cmap='RdYlGn', 
            center=0.5, vmin=0, vmax=1, cbar_kws={'label': 'Score'},
            linewidths=2, linecolor='white', ax=ax, annot_kws={'fontsize': 11, 'fontweight': 'bold'})

ax.set_xlabel('Performance Metric', fontsize=13, fontweight='bold')
ax.set_ylabel('Dataset Split', fontsize=13, fontweight='bold')
ax.set_title('Performance Heatmap: Cardiotoxicity Prediction Model\n(RFE-Selected Features, n=40)',
             fontsize=15, fontweight='bold', pad=20)
ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=12)
ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=11)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '31_performance_heatmap_documentation.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Heatmap saved: 31_performance_heatmap_documentation.png")

# ============================================================================
# 8. Hyperparameter Documentation
# ============================================================================
print("\n" + "="*80)
print("8. HYPERPARAMETER DOCUMENTATION")
print("="*80)

# Extract hyperparameters from model card
hyperparams = model_card['hyperparameters']

# Create dataframe
hyperparams_df = pd.DataFrame({
    'Parameter': list(hyperparams.keys()),
    'Value': list(hyperparams.values()),
    'Type': [type(v).__name__ for v in hyperparams.values()]
})

# Add descriptions
param_descriptions = {
    'colsample_bylevel': 'Subsample ratio of columns for each level',
    'colsample_bynode': 'Subsample ratio of columns for each node split',
    'colsample_bytree': 'Subsample ratio of columns when constructing each tree',
    'gamma': 'Minimum loss reduction for a split (regularization)',
    'learning_rate': 'Step size shrinkage to prevent overfitting',
    'max_delta_step': 'Maximum delta step for weight estimation (helps imbalanced classes)',
    'max_depth': 'Maximum tree depth',
    'max_leaves': 'Maximum number of leaves in a tree',
    'min_child_weight': 'Minimum sum of instance weight needed in a child',
    'n_estimators': 'Number of boosting rounds (trees)',
    'reg_alpha': 'L1 regularization term on weights',
    'reg_lambda': 'L2 regularization term on weights',
    'scale_pos_weight': 'Balancing of positive and negative weights',
    'subsample': 'Subsample ratio of training instances'
}

hyperparams_df['Description'] = hyperparams_df['Parameter'].map(param_descriptions)

# Save to CSV
hyperparams_df.to_csv(TABLES_DIR / 'optimal_hyperparameters.csv', index=False)

print("\n   Optimal Hyperparameters:")
print(hyperparams_df.to_string(index=False))
print(f"\n   ✓ Table saved: optimal_hyperparameters.csv")

# Create visualization
fig, ax = plt.subplots(figsize=(12, 8))

# Group parameters by category
structure_params = ['max_depth', 'max_leaves', 'min_child_weight']
sampling_params = ['subsample', 'colsample_bytree', 'colsample_bylevel', 'colsample_bynode']
learning_params = ['learning_rate', 'n_estimators']
regularization_params = ['reg_alpha', 'reg_lambda', 'gamma']
imbalance_params = ['scale_pos_weight', 'max_delta_step']

categories = {
    'Tree Structure': structure_params,
    'Sampling': sampling_params,
    'Learning': learning_params,
    'Regularization': regularization_params,
    'Class Imbalance': imbalance_params
}

y_pos = 0
colors_cat = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6']
yticks = []
ytick_labels = []

for (cat_name, params), color in zip(categories.items(), colors_cat):
    for param in params:
        value = hyperparams[param]
        # Normalize values for visualization (log scale for some)
        if param in ['n_estimators', 'reg_alpha', 'reg_lambda']:
            display_value = np.log10(value + 1) if value > 0 else 0
        else:
            display_value = value
        
        ax.barh(y_pos, display_value, color=color, alpha=0.7, edgecolor='black', linewidth=1.5)
        ax.text(display_value + 0.1, y_pos, f'{value:.4g}', 
                va='center', fontsize=10, fontweight='bold')
        yticks.append(y_pos)
        ytick_labels.append(param)
        y_pos += 1
    y_pos += 0.5  # Space between categories

ax.set_yticks(yticks)
ax.set_yticklabels(ytick_labels, fontsize=11)
ax.set_xlabel('Parameter Value (scaled)', fontsize=13, fontweight='bold')
ax.set_title('Optimal Hyperparameters: Cardiotoxicity Prediction Model\n(RFE-Selected Features, n=40)',
             fontsize=15, fontweight='bold', pad=20)
ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_facecolor('#f8f9fa')

# Add legend
from matplotlib.patches import Patch
legend_elements = [Patch(facecolor=color, alpha=0.7, edgecolor='black', label=cat)
                   for cat, color in zip(categories.keys(), colors_cat)]
ax.legend(handles=legend_elements, loc='lower right', fontsize=10, frameon=True, shadow=True)

plt.tight_layout()
plt.savefig(FIGURES_DIR / '32_hyperparameters_visualization.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Hyperparameters visualization saved: 32_hyperparameters_visualization.png")

# ============================================================================
# 9. Feature Importance Rankings Table
# ============================================================================
print("\n" + "="*80)
print("9. FEATURE IMPORTANCE RANKINGS")
print("="*80)

# Load feature importance
importance_df = pd.read_csv(MODELS_DIR / 'final_best_feature_importance.csv')

# Add rank and percentage
importance_df['Rank'] = range(1, len(importance_df) + 1)
total_importance = importance_df['importance'].sum()
importance_df['Importance (%)'] = (importance_df['importance'] / total_importance * 100).round(2)
importance_df['Cumulative (%)'] = importance_df['Importance (%)'].cumsum().round(2)

# Reorder columns
importance_df = importance_df[['Rank', 'feature', 'importance', 'Importance (%)', 'Cumulative (%)']]
importance_df.columns = ['Rank', 'Feature', 'Importance Score', 'Importance (%)', 'Cumulative (%)']

# Save full table
importance_df.to_csv(TABLES_DIR / 'feature_importance_rankings.csv', index=False)

print("\n   Top 15 Features by Importance:")
print(importance_df.head(15).to_string(index=False))
print(f"\n   ✓ Full table saved: feature_importance_rankings.csv")

# Visualize top 20 features
fig, ax = plt.subplots(figsize=(12, 10))

top_20 = importance_df.head(20)

# Create horizontal bar chart with gradient colors
bars = ax.barh(range(len(top_20)), top_20['Importance Score'], 
               color=plt.cm.viridis(np.linspace(0.3, 0.9, len(top_20))),
               edgecolor='black', linewidth=1.2)

ax.set_yticks(range(len(top_20)))
ax.set_yticklabels(top_20['Feature'], fontsize=11)
ax.invert_yaxis()
ax.set_xlabel('Feature Importance Score', fontsize=13, fontweight='bold')
ax.set_title('Top 20 Feature Importance Rankings\n(RFE-Selected Features, n=40)',
             fontsize=15, fontweight='bold', pad=20)

# Add value labels
for i, (score, pct) in enumerate(zip(top_20['Importance Score'], top_20['Importance (%)'])):
    ax.text(score + 0.002, i, f'{score:.4f} ({pct:.1f}%)', 
            va='center', fontsize=9, fontweight='bold')

ax.grid(True, axis='x', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_facecolor('#f8f9fa')
fig.patch.set_facecolor('white')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '33_feature_importance_rankings.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Feature importance visualization saved: 33_feature_importance_rankings.png")

# ============================================================================
# 10. Feature Categories Analysis
# ============================================================================
print("\n" + "="*80)
print("10. FEATURE CATEGORIES ANALYSIS")
print("="*80)

# Load feature categories
with open(ENGINEERED_DIR / 'feature_categories.json', 'r') as f:
    feature_categories = json.load(f)

# Create reverse mapping (feature -> category)
feature_to_category = {}
for category, features_list in feature_categories.items():
    for feature in features_list:
        feature_to_category[feature] = category

# Add category to importance dataframe
importance_df['Category'] = importance_df['Feature'].map(feature_to_category)

# Calculate category-level importance
category_importance = importance_df.groupby('Category').agg({
    'Importance Score': 'sum',
    'Feature': 'count'
}).reset_index()
category_importance.columns = ['Category', 'Total Importance', 'Feature Count']
category_importance['Avg Importance'] = (category_importance['Total Importance'] / 
                                         category_importance['Feature Count'])
category_importance = category_importance.sort_values('Total Importance', ascending=False)

# Save table
category_importance.to_csv(TABLES_DIR / 'category_importance.csv', index=False)

print("\n   Feature Category Importance:")
print(category_importance.to_string(index=False))
print(f"\n   ✓ Table saved: category_importance.csv")

# Visualize categories
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))

# Total importance by category
colors_pie = plt.cm.Set3(range(len(category_importance)))
bars1 = ax1.barh(category_importance['Category'], category_importance['Total Importance'],
                 color=colors_pie, edgecolor='black', linewidth=1.5)
ax1.set_xlabel('Total Importance', fontsize=13, fontweight='bold')
ax1.set_title('Total Feature Importance by Category', fontsize=14, fontweight='bold', pad=15)
ax1.grid(True, axis='x', alpha=0.3)
ax1.set_facecolor('#f8f9fa')

# Add value labels
for i, (cat, val) in enumerate(zip(category_importance['Category'], 
                                    category_importance['Total Importance'])):
    ax1.text(val + 0.01, i, f'{val:.3f}', va='center', fontsize=10, fontweight='bold')

# Average importance by category
bars2 = ax2.barh(category_importance['Category'], category_importance['Avg Importance'],
                 color=colors_pie, edgecolor='black', linewidth=1.5)
ax2.set_xlabel('Average Importance per Feature', fontsize=13, fontweight='bold')
ax2.set_title('Average Feature Importance by Category', fontsize=14, fontweight='bold', pad=15)
ax2.grid(True, axis='x', alpha=0.3)
ax2.set_facecolor('#f8f9fa')

# Add value labels
for i, (cat, val) in enumerate(zip(category_importance['Category'], 
                                    category_importance['Avg Importance'])):
    ax2.text(val + 0.002, i, f'{val:.4f}', va='center', fontsize=10, fontweight='bold')

plt.suptitle('Feature Category Analysis: Cardiotoxicity Prediction Model\n(RFE-Selected Features, n=40)',
             fontsize=16, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig(FIGURES_DIR / '34_category_importance.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Category importance visualization saved: 34_category_importance.png")

# ============================================================================
# 11. Model Comparison Summary
# ============================================================================
print("\n" + "="*80)
print("11. MODEL COMPARISON SUMMARY")
print("="*80)

# Load baseline and optimization results for comparison
with open(MODELS_DIR / 'baseline_metrics.json', 'r') as f:
    baseline_metrics = json.load(f)

with open(MODELS_DIR / 'random_search_metrics.json', 'r') as f:
    random_search_metrics = json.load(f)

# Create comparison table
comparison_data = {
    'Model Version': ['Baseline', 'Phase 7 (Random Search)', 'Phase 8 (RFE Selected - Final)'],
    'Features': [88, 88, 40],
    'Test ROC AUC': [
        baseline_metrics.get('test_roc_auc', 0.6094),
        random_search_metrics.get('test_roc_auc', 0.7604),
        model_card['performance']['test']['roc_auc']
    ],
    'Test PR AUC': [
        baseline_metrics.get('test_pr_auc', 0.3489),
        random_search_metrics.get('test_pr_auc', 0.3686),
        model_card['performance']['test']['pr_auc']
    ],
    'Val ROC AUC': [
        baseline_metrics.get('val_roc_auc', 0.6111),
        random_search_metrics.get('val_roc_auc', 0.7049),
        model_card['performance']['validation']['roc_auc']
    ]
}

comparison_df = pd.DataFrame(comparison_data)
comparison_df['Improvement vs Baseline (%)'] = (
    (comparison_df['Test ROC AUC'] - comparison_df['Test ROC AUC'].iloc[0]) / 
    comparison_df['Test ROC AUC'].iloc[0] * 100
).round(2)

# Save table
comparison_df.to_csv(TABLES_DIR / 'model_progression_comparison.csv', index=False)

print("\n   Model Progression Comparison:")
print(comparison_df.to_string(index=False))
print(f"\n   ✓ Table saved: model_progression_comparison.csv")

# Visualize progression
fig, ax = plt.subplots(figsize=(12, 7))

x = np.arange(len(comparison_df))
width = 0.25

bars1 = ax.bar(x - width, comparison_df['Test ROC AUC'], width, 
               label='Test ROC AUC', color='#F18F01', edgecolor='black', linewidth=1.5)
bars2 = ax.bar(x, comparison_df['Val ROC AUC'], width, 
               label='Val ROC AUC', color='#A23B72', edgecolor='black', linewidth=1.5)
bars3 = ax.bar(x + width, comparison_df['Test PR AUC'], width, 
               label='Test PR AUC', color='#2E86AB', edgecolor='black', linewidth=1.5)

ax.set_xlabel('Model Version', fontsize=13, fontweight='bold')
ax.set_ylabel('Score', fontsize=13, fontweight='bold')
ax.set_title('Model Development Progression: Baseline → Optimized → Final\n(Cardiotoxicity Prediction)',
             fontsize=15, fontweight='bold', pad=20)
ax.set_xticks(x)
ax.set_xticklabels(comparison_df['Model Version'], fontsize=11, rotation=15, ha='right')
ax.legend(fontsize=11, frameon=True, shadow=True, loc='upper left')
ax.grid(True, axis='y', alpha=0.3, linestyle='-', linewidth=0.5)
ax.set_ylim([0, 1])
ax.set_facecolor('#f8f9fa')

# Add value labels on bars
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height:.4f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

plt.tight_layout()
plt.savefig(FIGURES_DIR / '35_model_progression.png', dpi=300, bbox_inches='tight')
plt.close()

print(f"   ✓ Model progression visualization saved: 35_model_progression.png")

# ============================================================================
# 12. Final Documentation Report
# ============================================================================
print("\n" + "="*80)
print("12. GENERATING FINAL DOCUMENTATION REPORT")
print("="*80)

report = {
    'report_title': 'Cardiotoxicity Prediction Model - Final Documentation',
    'model_name': model_card['model_name'],
    'model_version': model_card['version'],
    'date_generated': '2025-10-06',
    'phase': 'Phase 10: Model Documentation',
    
    'model_specifications': {
        'feature_selection_method': model_card['feature_selection_method'],
        'feature_set': model_card['feature_set'],
        'n_features': model_card['n_features'],
        'algorithm': 'XGBoost (Extreme Gradient Boosting)',
        'optimization_target': 'ROC AUC',
        'class_imbalance_handling': 'scale_pos_weight + max_delta_step'
    },
    
    'dataset_information': {
        'total_samples': len(train_df) + len(val_df) + len(test_df),
        'train_samples': len(train_df),
        'validation_samples': len(val_df),
        'test_samples': len(test_df),
        'positive_class_ratio': f"{(y_test.sum() / len(y_test) * 100):.2f}%",
        'class_imbalance_ratio': f"1:{(len(y_test) - y_test.sum()) / y_test.sum():.1f}"
    },
    
    'performance_summary': {
        'train': {
            'roc_auc': float(train_metrics['ROC AUC']),
            'pr_auc': float(train_metrics['PR AUC']),
            'accuracy': float(train_metrics['Accuracy']),
            'balanced_accuracy': float(train_metrics['Balanced Accuracy']),
            'sensitivity': float(train_metrics['Sensitivity (Recall)']),
            'specificity': float(train_metrics['Specificity'])
        },
        'validation': {
            'roc_auc': float(val_metrics['ROC AUC']),
            'pr_auc': float(val_metrics['PR AUC']),
            'accuracy': float(val_metrics['Accuracy']),
            'balanced_accuracy': float(val_metrics['Balanced Accuracy']),
            'sensitivity': float(val_metrics['Sensitivity (Recall)']),
            'specificity': float(val_metrics['Specificity'])
        },
        'test': {
            'roc_auc': float(test_metrics['ROC AUC']),
            'pr_auc': float(test_metrics['PR AUC']),
            'accuracy': float(test_metrics['Accuracy']),
            'balanced_accuracy': float(test_metrics['Balanced Accuracy']),
            'sensitivity': float(test_metrics['Sensitivity (Recall)']),
            'specificity': float(test_metrics['Specificity'])
        }
    },
    
    'model_progression': {
        'baseline_test_auc': float(comparison_df.iloc[0]['Test ROC AUC']),
        'phase7_test_auc': float(comparison_df.iloc[1]['Test ROC AUC']),
        'final_test_auc': float(comparison_df.iloc[2]['Test ROC AUC']),
        'total_improvement_pct': float(comparison_df.iloc[2]['Improvement vs Baseline (%)']),
        'feature_reduction': f"{88} → {40} features ({((88-40)/88*100):.1f}% reduction)"
    },
    
    'top_features': model_card['top_features'][:10],
    
    'hyperparameters': model_card['hyperparameters'],
    
    'outputs_generated': {
        'visualizations': [
            '28_final_roc_curves_documentation.png',
            '29_final_pr_curves_documentation.png',
            '30_final_confusion_matrices_documentation.png',
            '31_performance_heatmap_documentation.png',
            '32_hyperparameters_visualization.png',
            '33_feature_importance_rankings.png',
            '34_category_importance.png',
            '35_model_progression.png'
        ],
        'tables': [
            'performance_metrics_comparison.csv',
            'optimal_hyperparameters.csv',
            'feature_importance_rankings.csv',
            'category_importance.csv',
            'model_progression_comparison.csv'
        ]
    },
    
    'key_findings': [
        f"Best model achieved Test ROC AUC of {model_card['performance']['test']['roc_auc']:.4f}",
        f"RFE feature selection improved performance by {model_card['comparison_with_phase7']['improvement']:.2f}%",
        f"Feature reduction from 88 to 40 features (54.5% fewer) improved generalization",
        f"Top predictor: {model_card['top_features'][0]['feature']} (importance: {model_card['top_features'][0]['importance']:.4f})",
        f"Overall improvement from baseline: {comparison_df.iloc[2]['Improvement vs Baseline (%)']:.2f}%"
    ]
}

# Save report
with open(TABLES_DIR / 'final_documentation_report.json', 'w') as f:
    json.dump(report, f, indent=2)

print("\n   Final Documentation Report:")
print(f"   • Model: {report['model_specifications']['algorithm']}")
print(f"   • Features: {report['model_specifications']['n_features']} (RFE-selected)")
print(f"   • Test ROC AUC: {report['performance_summary']['test']['roc_auc']:.4f}")
print(f"   • Test PR AUC: {report['performance_summary']['test']['pr_auc']:.4f}")
print(f"   • Improvement vs Baseline: +{report['model_progression']['total_improvement_pct']:.2f}%")
print(f"\n   ✓ Report saved: final_documentation_report.json")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*80)
print("PHASE 10: MODEL DOCUMENTATION - COMPLETED")
print("="*80)

print(f"\n📊 Visualizations Generated: {len(report['outputs_generated']['visualizations'])}")
for viz in report['outputs_generated']['visualizations']:
    print(f"   • {viz}")

print(f"\n📋 Tables Generated: {len(report['outputs_generated']['tables'])}")
for table in report['outputs_generated']['tables']:
    print(f"   • {table}")

print(f"\n🎯 Key Performance Metrics (Test Set):")
print(f"   • ROC AUC: {test_metrics['ROC AUC']:.4f}")
print(f"   • PR AUC: {test_metrics['PR AUC']:.4f}")
print(f"   • Balanced Accuracy: {test_metrics['Balanced Accuracy']:.4f}")
print(f"   • Sensitivity: {test_metrics['Sensitivity (Recall)']:.4f}")
print(f"   • Specificity: {test_metrics['Specificity']:.4f}")

print(f"\n✨ Model Highlights:")
print(f"   • Best feature selection: RFE (40 features)")
print(f"   • Feature reduction: 88 → 40 (54.5% fewer)")
print(f"   • Improvement over baseline: +{report['model_progression']['total_improvement_pct']:.2f}%")
print(f"   • Top predictor: {model_card['top_features'][0]['feature']}")

print("\n" + "="*80)
print("All documentation files saved to:")
print(f"   Figures: {FIGURES_DIR}")
print(f"   Tables: {TABLES_DIR}")
print("="*80)

