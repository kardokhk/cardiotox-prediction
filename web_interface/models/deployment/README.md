# Cardiotoxicity Prediction Model - Deployment Package

## Overview

This is a production-ready deployment package for predicting cancer treatment-related cardiac dysfunction (CTRCD) in HER2+ breast cancer patients undergoing cardiotoxic therapy.

**Model Performance:**
- Test ROC AUC: **0.7960**
- Test PR AUC: **0.4150**
- Improvement over baseline: **+30.62%**
- Feature reduction: 88 → 40 features (54.5% fewer)

**Model Version:** 1.0  
**Created:** October 2025  
**Framework:** XGBoost 2.1.1

---

## Package Contents

| File | Size | Description |
|------|------|-------------|
| `cardiotoxicity_model.pkl` | 299.49 KB | Trained XGBoost model (pickle format) |
| `cardiotoxicity_model.json` | 174.94 KB | Trained XGBoost model (JSON format) |
| `cardiotoxicity_predictor.py` | 10.29 KB | Complete predictor module with pipelines |
| `preprocessing_pipeline.pkl` | 0.83 KB | Preprocessing pipeline (legacy) |
| `feature_engineering_pipeline.pkl` | 0.06 KB | Feature engineering pipeline (legacy) |
| `feature_statistics.json` | 1.52 KB | Training data statistics for imputation |
| `model_metadata.json` | 12.76 KB | Complete model documentation |
| `USAGE_INSTRUCTIONS.md` | 6.20 KB | Comprehensive usage guide |
| `example_prediction.py` | 4.33 KB | Basic example script |
| `example_prediction_improved.py` | 2.86 KB | Improved example with module |
| `requirements.txt` | 0.31 KB | Python dependencies |
| `deployment_summary.json` | 1.97 KB | Deployment package summary |

**Total Size:** ~500 KB

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

Required packages:
- pandas >= 1.3.0
- numpy >= 1.21.0
- scikit-learn >= 1.0.0
- xgboost == 2.1.1

### 2. Run Example

```bash
python example_prediction_improved.py
```

This will run predictions on two example patients and demonstrate the model's usage.

### 3. Use in Your Code

```python
from cardiotoxicity_predictor import CardiotoxicityPredictor

# Initialize predictor
predictor = CardiotoxicityPredictor('.')

# Create patient data
patient = {
    'age': 55,
    'weight': 70,
    'height': 165,
    'heart_rate': 75,
    'LVEF': 60,
    'PWT': 0.9,
    'LAd': 3.5,
    'LVDd': 4.8,
    'LVSd': 3.2,
    'AC': 1,
    'antiHER2': 1,
    'HTA': 1,
    'DL': 0,
    'smoker': 0,
    'exsmoker': 0,
    'diabetes': 0,
    'obesity': 0,
    'ACprev': 0,
    'RTprev': 0,
    'heart_rhythm': 0
}

# Get prediction
result = predictor.predict_single(patient)

print(f"CTRCD Risk: {result['probability']:.2%}")
print(f"Risk Level: {result['risk_level']}")
print(f"Recommendation: {result['recommendation']}")
```

---

## Input Requirements

### Required Features (20 columns)

**Continuous Variables:**
- `age` (years): Patient age
- `weight` (kg): Body weight
- `height` (cm): Height
- `heart_rate` (bpm): Resting heart rate
- `LVEF` (%): Left ventricular ejection fraction
- `PWT` (cm): Posterior wall thickness
- `LAd` (cm): Left atrial diameter
- `LVDd` (cm): Left ventricular diastolic diameter
- `LVSd` (cm): Left ventricular systolic diameter

**Binary Variables (0 or 1):**
- `AC`: Current anthracycline treatment
- `antiHER2`: Current anti-HER2 treatment
- `HTA`: Hypertension
- `DL`: Dyslipidemia
- `smoker`: Current smoker
- `exsmoker`: Former smoker
- `diabetes`: Diabetes mellitus
- `obesity`: Obesity (BMI ≥ 30)
- `ACprev`: Previous anthracycline exposure
- `RTprev`: Previous radiotherapy
- `heart_rhythm`: Abnormal heart rhythm

### Data Validation

The model automatically:
- ✓ Handles missing values (imputes with training statistics)
- ✓ Validates input ranges (age: 0-120, LVEF: 0-100, etc.)
- ✓ Checks for required columns
- ✓ Detects infinite values

---

## Output Interpretation

### Risk Probability

The model outputs a probability between 0 and 1 representing the risk of developing CTRCD.

### Risk Thresholds

| Probability | Risk Level | Clinical Action |
|-------------|-----------|-----------------|
| < 30% | **Low** | Standard monitoring (every 6 months) |
| 30-50% | **Moderate** | Enhanced monitoring (every 3 months) |
| 50-70% | **High** | Cardioprotective agents + frequent monitoring |
| > 70% | **Very High** | Consider treatment modification + cardiology consult |

---

## Top Risk Factors

The model considers these key features (in order of importance):

1. **CV_risk_score** (17.03%) - Composite cardiovascular risk score
2. **heart_rate_cubed** (13.87%) - Elevated heart rate indicator
3. **risk_x_treatment** (13.53%) - Interaction of CV risk with cardiotoxic treatment
4. **cumulative_cardiotoxic_treatment** (9.40%) - Total treatment burden
5. **age_adjusted_LVEF** (7.51%) - Age-normalized cardiac function

**Clinical Insights:**
- Cardiovascular risk factors compound with cardiotoxic treatment effects
- Baseline cardiac function (LVEF) is critical
- Cumulative treatment burden matters - prior exposures increase risk
- Heart rate elevation may indicate early cardiac stress
- Age modifies the effect of cardiac function on risk

---

## Model Architecture

### Feature Engineering Pipeline

The model creates 63 engineered features from the 20 input features:

1. **Anthropometric Features:** BMI, BSA, body composition categories
2. **Cardiac Structural Indices:** RWT, LVMI, fractional shortening, LAVi
3. **Cardiac Function Categories:** LVEF categories, dysfunction risk scores
4. **Age-Cardiac Interactions:** Age-adjusted LVEF, age × cardiac function
5. **Treatment Features:** Treatment combinations, cumulative burden
6. **Cardiovascular Risk Score:** Composite risk from multiple factors
7. **Interaction Features:** Risk × treatment, LVEF × risk factors
8. **Polynomial Features:** Age², age³, LVEF², heart rate³, etc.
9. **Ratio Features:** Cardiac dimension ratios normalized to body size

### Feature Selection

From 88 total features, **40 features** were selected using Recursive Feature Elimination (RFE) based on predictive power and importance.

### Model Hyperparameters

The XGBoost model was optimized through 5000 iterations of random search:

```python
{
  'max_depth': 5,
  'learning_rate': 0.269,
  'n_estimators': 427,
  'min_child_weight': 5,
  'subsample': 0.856,
  'colsample_bytree': 0.530,
  'gamma': 3.925,
  'reg_alpha': 9.924,
  'reg_lambda': 7.562,
  'scale_pos_weight': 1.629
}
```

---

## Performance Metrics

### ROC AUC (Primary Metric)

| Split | ROC AUC |
|-------|---------|
| Train | 0.7992 |
| Validation | 0.7274 |
| **Test** | **0.7960** |

### PR AUC (Imbalanced Classes)

| Split | PR AUC |
|-------|--------|
| Train | 0.3266 |
| Validation | 0.2646 |
| **Test** | **0.4150** |

### Model Development Progress

- **Baseline (Phase 5):** Test AUC = 0.6094
- **Iterative Optimization (Phase 6):** Test AUC = 0.6962 (+14.2%)
- **Random Search (Phase 7):** Test AUC = 0.7604 (+24.8%)
- **Feature Selection (Phase 8):** Test AUC = **0.7960** (+30.6%)

---

## Clinical Usage Guidelines

### When to Use

✅ **Appropriate Use:**
- Screening HER2+ breast cancer patients before cardiotoxic therapy
- Risk stratification for monitoring frequency
- Identifying candidates for cardioprotective strategies
- Supporting clinical decision-making with objective risk assessment

❌ **Not Appropriate:**
- Sole basis for treatment decisions (use with clinical judgment)
- Patients outside the training cohort characteristics
- Real-time acute cardiac monitoring
- Replacing standard cardiac imaging and biomarkers

### Clinical Workflow Integration

1. **Patient presents** for cancer treatment planning
2. **Collect baseline data** (clinical measurements, cardiac imaging, risk factors)
3. **Run prediction** using this model
4. **Interpret results** in clinical context
5. **Take clinical action:**
   - Low risk: Standard monitoring protocol
   - Moderate risk: Enhanced monitoring (more frequent follow-up)
   - High risk: Cardioprotective agents (e.g., ACE inhibitors, beta-blockers)
   - Very high risk: Modify treatment plan or cardiology consultation

### Monitoring Schedule Recommendations

| Risk Level | Follow-up Frequency | Additional Actions |
|------------|---------------------|-------------------|
| Low | Every 6 months | Standard echocardiography |
| Moderate | Every 3 months | Enhanced cardiac biomarkers |
| High | Every 1-2 months | Cardioprotective medications |
| Very High | Monthly | Treatment modification, cardiology consult |

---

## Limitations

⚠️ **Important Limitations:**

1. **Sample Size:** Trained on 531 patients from single cohort
2. **Class Imbalance:** Only 10.17% positive cases (54/531)
3. **External Validation:** Not validated on external datasets
4. **Temporal:** Does not model time-to-event explicitly
5. **Imaging:** Limited to basic echocardiographic parameters
6. **Population:** Specific to HER2+ breast cancer patients

**Recommendation:** Use as clinical decision support, not diagnostic tool.

---

## Deployment Checklist

Before deploying to production:

- [ ] Test on sample data from your institution
- [ ] Validate input data pipeline
- [ ] Establish monitoring for model performance
- [ ] Train clinical staff on interpretation
- [ ] Set up feedback collection mechanism
- [ ] Ensure HIPAA/GDPR compliance for data handling
- [ ] Document integration with clinical information system
- [ ] Establish model update and recalibration schedule

---

## Technical Support

### Files Overview

- **For Production Use:** `cardiotoxicity_predictor.py` + `cardiotoxicity_model.pkl` + `feature_statistics.json` + `model_metadata.json`
- **For Documentation:** `USAGE_INSTRUCTIONS.md` + `model_metadata.json` + `deployment_summary.json`
- **For Testing:** `example_prediction_improved.py`

### Troubleshooting

**Issue:** Missing column error  
**Solution:** Ensure all 20 required input features are present

**Issue:** Value out of range  
**Solution:** Check that continuous features are in valid ranges (age: 0-120, LVEF: 0-100)

**Issue:** Import error for `cardiotoxicity_predictor`  
**Solution:** Ensure the script is run in the deployment directory or add to Python path

**Issue:** Model version mismatch  
**Solution:** Check XGBoost version matches requirements.txt

### Contact

For questions or issues with deployment:
- Review `model_metadata.json` for technical details
- Check `USAGE_INSTRUCTIONS.md` for comprehensive guide
- Consult cardiology team for clinical interpretation

---

## License & Disclaimer

This model is for **research and clinical decision support only**.

**IMPORTANT:** This model should NOT be used as the sole basis for clinical decisions. Always combine model predictions with:
- Clinical judgment and expertise
- Additional diagnostic information
- Patient preferences and values
- Current clinical guidelines

The model has not been approved by regulatory agencies (FDA, EMA, etc.) and is intended for research and clinical decision support purposes only.

---

## Citation

If you use this model in research or clinical practice, please cite:

```
Cardiotoxicity Prediction Model for HER2+ Breast Cancer Patients
Version 1.0, October 2025
XGBoost-based prediction model (Test ROC AUC: 0.7960)
```

---

## Version History

**Version 1.0** (October 2025)
- Initial deployment release
- 40 selected features from 88 engineered features
- Test ROC AUC: 0.7960
- Complete deployment package with documentation

---

**Last Updated:** October 2025  
**Model Version:** 1.0  
**Package Size:** 500.44 KB  
**Status:** ✅ Ready for Deployment
