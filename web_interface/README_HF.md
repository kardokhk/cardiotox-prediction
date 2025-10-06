---
title: CTRCD Risk Predictor
emoji: 🫀
colorFrom: blue
colorTo: purple
sdk: gradio
sdk_version: 4.7.1
app_file: app.py
pinned: false
license: mit
short_description: Advanced ML model for predicting cancer treatment-related cardiac dysfunction in HER2+ breast cancer patients
tags:
- machine-learning
- healthcare
- cardiology
- oncology
- xgboost
- medical-ai
- clinical-decision-support
---

# 🫀 CTRCD Risk Predictor

> **Advanced Machine Learning for Cardiotoxicity Prediction in HER2+ Breast Cancer Patients**

## 🎯 Overview

This application provides **real-time risk assessment** for Cancer Treatment-Related Cardiac Dysfunction (CTRCD) in HER2+ breast cancer patients. Using a state-of-the-art XGBoost model trained on 531 patients, it achieves **79.6% ROC AUC** with a **30.6% improvement** over baseline models.

## ✨ Key Features

- **🏥 Clinical-Grade Interface**: Professional design suitable for healthcare settings
- **⚡ Real-Time Predictions**: Instant risk assessment with comprehensive recommendations
- **🎯 High Accuracy**: Test ROC AUC of 0.796, validated on independent test set
- **📊 Risk Stratification**: Clear low/moderate/high/very high risk categories
- **🔬 Evidence-Based**: 40 carefully selected clinical and cardiac parameters

## 🩺 Clinical Impact

### Risk Categories
- **🟢 Low (<30%)**: Standard 3-6 month cardiac monitoring
- **🟡 Moderate (30-50%)**: Enhanced monitoring every 3 months  
- **🟠 High (50-70%)**: Cardioprotective agents + frequent monitoring
- **🔴 Very High (>70%)**: Consider treatment modification + cardiology consult

## 📊 Model Performance

| Metric | Training | Validation | Test |
|--------|----------|------------|------|
| **ROC AUC** | 0.799 | 0.727 | **0.796** |
| **PR AUC** | 0.327 | 0.265 | 0.415 |

## 🚀 Usage

1. **Enter Patient Data**: Use intuitive sliders and checkboxes
2. **Calculate Risk**: Click "Calculate CTRCD Risk"  
3. **Review Results**: Get probability, risk level, and clinical recommendations
4. **Clinical Action**: Follow evidence-based monitoring guidelines

## ⚠️ Important Disclaimers

- **Clinical Decision Support Only**: Not intended as sole basis for medical decisions
- **Validated Population**: HER2+ breast cancer patients undergoing cardiotoxic therapy
- **Professional Use**: Combine with clinical judgment and additional diagnostics