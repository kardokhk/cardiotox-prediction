# 🫀 CTRCD Risk Predictor

> **Advanced Machine Learning for Cardiotoxicity Prediction in HER2+ Breast Cancer Patients**

[![Model Performance](https://img.shields.io/badge/Test%20ROC%20AUC-0.796-brightgreen.svg)](https://github.com/your-username/your-repo)
[![Framework](https://img.shields.io/badge/Framework-XGBoost-orange.svg)](https://xgboost.readthedocs.io/)
[![Dataset](https://img.shields.io/badge/Patients-531-blue.svg)](https://github.com/your-username/your-repo)
[![Improvement](https://img.shields.io/badge/Baseline%20Improvement-+30.6%25-success.svg)](https://github.com/your-username/your-repo)

## 🎯 Overview

This application provides **real-time risk assessment** for Cancer Treatment-Related Cardiac Dysfunction (CTRCD) in HER2+ breast cancer patients. Using a state-of-the-art XGBoost model trained on 531 patients, it achieves **79.6% ROC AUC** with a **30.6% improvement** over baseline models.

### 🚀 Quick Start

#### For Hugging Face Spaces Deployment:

1. Create a new Space on Hugging Face
2. Choose "Gradio" as the SDK
3. Upload all files from this directory
4. The app will automatically deploy!

#### For Local Development:

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
python app.py
```

Visit `http://localhost:7860` to access the interface.

### 📊 Model Performance

- **Test ROC AUC:** 0.7960 (+30.6% improvement over baseline)
- **Features:** 40 carefully selected clinical and cardiac parameters
- **Framework:** XGBoost 2.1.1
- **Training Data:** 531 HER2+ breast cancer patients

### 🎯 Features

- **Interactive Web Interface:** Easy-to-use form for entering patient data
- **Real-time Predictions:** Instant risk assessment with detailed recommendations
- **Clinical Context:** Risk levels with actionable clinical guidance
- **Professional Design:** Clean, medical-grade interface suitable for clinical settings
- **Example Cases:** Pre-loaded examples for demonstration

### 📝 Input Parameters

#### Demographics
- Age, Weight, Height

#### Cardiac Parameters
- Heart Rate, LVEF (Left Ventricular Ejection Fraction)
- PWT (Posterior Wall Thickness), LAd (Left Atrial Diameter)
- LVDd/LVSd (Left Ventricular Diameters)

#### Treatment Profile
- Current Anthracycline/Anti-HER2 therapy
- Previous cardiotoxic treatments

#### Risk Factors
- Hypertension, Diabetes, Dyslipidemia
- Smoking history, Obesity, Arrhythmias

### 🎨 Risk Levels

- 🟢 **Low** (<30%): Standard 3-6 month follow-up
- 🟡 **Moderate** (30-50%): Enhanced monitoring every 3 months
- 🟠 **High** (50-70%): Cardioprotective agents + frequent monitoring
- 🔴 **Very High** (>70%): Consider treatment modification + cardiology consult

### ⚠️ Important Notes

- This model is for **research and clinical decision support only**
- Always combine predictions with clinical judgment
- Not intended as the sole basis for clinical decisions
- Validated on HER2+ breast cancer patients

### 🔧 Technical Details

- **Framework:** Gradio 4.7.1
- **Model:** XGBoost Classifier
- **Input Validation:** Automatic handling of missing values
- **Feature Engineering:** 40 optimally selected features from 88 engineered
- **Deployment Ready:** Compatible with Hugging Face Spaces

### 📚 Citation

If you use this model in your research, please cite:

```
Cardiotoxicity Prediction Model for HER2+ Breast Cancer Patients
Version 1.0, October 2025
XGBoost Classifier, Test ROC AUC: 0.796
```

### 🆘 Support

For technical issues or questions about the model:
- Review the model metadata in `models/deployment/model_metadata.json`
- Check the comprehensive usage guide in `models/deployment/USAGE_INSTRUCTIONS.md`
- Consult with your cardiology team for clinical interpretation

---

**Last Updated:** October 2025  
**Model Version:** 1.0  
**License:** For research and clinical decision support use only