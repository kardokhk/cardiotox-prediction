"""
Cardiotoxicity Predictor Module
Contains all necessary classes for model deployment
"""

import pickle
import pandas as pd
import numpy as np
import json
from pathlib import Path


class PreprocessingPipeline:
    """
    Preprocessing pipeline for cardiotoxicity prediction
    
    Steps:
    1. Handle missing values (median for continuous, mode for binary)
    2. Validate input data
    """
    
    def __init__(self, feature_stats):
        """
        Initialize with statistics from training data
        
        Args:
            feature_stats: Dictionary with median/mode values for imputation
        """
        self.feature_stats = feature_stats
        self.continuous_vars = ['age', 'weight', 'height', 'heart_rate', 
                               'LVEF', 'PWT', 'LAd', 'LVDd', 'LVSd']
        
    def preprocess(self, df):
        """
        Apply preprocessing to input dataframe
        
        Args:
            df: Input dataframe with raw features
            
        Returns:
            Preprocessed dataframe
        """
        df_processed = df.copy()
        
        # Handle missing values
        for col in df_processed.columns:
            if df_processed[col].isnull().sum() > 0:
                if col in self.continuous_vars:
                    # Use median for continuous variables
                    df_processed[col].fillna(self.feature_stats[col]['median'], inplace=True)
                else:
                    # Use mode for binary variables
                    df_processed[col].fillna(self.feature_stats[col]['mode'], inplace=True)
        
        return df_processed
    
    def validate(self, df):
        """
        Validate input dataframe
        
        Args:
            df: Input dataframe
            
        Returns:
            Tuple of (is_valid, error_message)
        """
        errors = []
        
        # Check for required columns
        required_cols = self.continuous_vars + [
            'AC', 'antiHER2', 'HTA', 'DL', 'smoker', 'exsmoker', 
            'diabetes', 'obesity', 'ACprev', 'RTprev', 'heart_rhythm'
        ]
        
        missing_cols = set(required_cols) - set(df.columns)
        if missing_cols:
            errors.append(f"Missing required columns: {missing_cols}")
        
        # Check for infinite values
        if np.isinf(df.select_dtypes(include=[np.number])).sum().sum() > 0:
            errors.append("Input contains infinite values")
        
        # Check for valid ranges
        if 'age' in df.columns:
            if (df['age'] < 0).any() or (df['age'] > 120).any():
                errors.append("Age values out of valid range (0-120)")
        
        if 'LVEF' in df.columns:
            if (df['LVEF'] < 0).any() or (df['LVEF'] > 100).any():
                errors.append("LVEF values out of valid range (0-100)")
        
        if errors:
            return False, "; ".join(errors)
        return True, "Valid"


class FeatureEngineeringPipeline:
    """
    Feature engineering pipeline for cardiotoxicity prediction
    
    Creates 63 engineered features from 25 original features
    """
    
    def __init__(self):
        pass
    
    def engineer_features(self, df):
        """
        Apply feature engineering to preprocessed dataframe
        
        Args:
            df: Preprocessed dataframe with original features
            
        Returns:
            Dataframe with engineered features
        """
        df_eng = df.copy()
        
        # 1. ANTHROPOMETRIC FEATURES
        df_eng['BMI'] = df_eng['weight'] / ((df_eng['height'] / 100) ** 2)
        df_eng['BMI_underweight'] = (df_eng['BMI'] < 18.5).astype(int)
        df_eng['BMI_normal'] = ((df_eng['BMI'] >= 18.5) & (df_eng['BMI'] < 25)).astype(int)
        df_eng['BMI_overweight'] = ((df_eng['BMI'] >= 25) & (df_eng['BMI'] < 30)).astype(int)
        df_eng['BMI_obese'] = (df_eng['BMI'] >= 30).astype(int)
        df_eng['BSA'] = np.sqrt((df_eng['height'] * df_eng['weight']) / 3600)
        
        # 2. CARDIAC STRUCTURAL INDICES
        df_eng['RWT'] = (2 * df_eng['PWT']) / df_eng['LVDd']
        df_eng['LV_mass_approx'] = 0.8 * (1.04 * ((df_eng['LVDd'] + 2*df_eng['PWT'])**3 - df_eng['LVDd']**3)) + 0.6
        df_eng['LVMI'] = df_eng['LV_mass_approx'] / df_eng['BSA']
        df_eng['Fractional_Shortening'] = ((df_eng['LVDd'] - df_eng['LVSd']) / df_eng['LVDd']) * 100
        df_eng['LA_volume_approx'] = (4/3) * np.pi * (df_eng['LAd']/2)**3
        df_eng['LAVi'] = df_eng['LA_volume_approx'] / df_eng['BSA']
        df_eng['LV_hypertrophy'] = (df_eng['PWT'] > 1.1).astype(int)
        df_eng['LA_enlargement'] = (df_eng['LAd'] > 4.0).astype(int)
        df_eng['LV_dilation'] = (df_eng['LVDd'] > 5.5).astype(int)
        
        # 3. CARDIAC FUNCTION CATEGORIES
        df_eng['LVEF_reduced'] = (df_eng['LVEF'] < 50).astype(int)
        df_eng['LVEF_midrange'] = ((df_eng['LVEF'] >= 50) & (df_eng['LVEF'] < 60)).astype(int)
        df_eng['LVEF_preserved'] = (df_eng['LVEF'] >= 60).astype(int)
        df_eng['LVEF_margin'] = df_eng['LVEF'] - 50
        df_eng['systolic_dysfunction_risk'] = (
            (df_eng['LVEF'] < 55).astype(int) + 
            (df_eng['Fractional_Shortening'] < 28).astype(int)
        )
        
        # 4. AGE-CARDIAC INTERACTIONS
        df_eng['age_group_young'] = (df_eng['age'] < 45).astype(int)
        df_eng['age_group_middle'] = ((df_eng['age'] >= 45) & (df_eng['age'] < 65)).astype(int)
        df_eng['age_group_elderly'] = (df_eng['age'] >= 65).astype(int)
        df_eng['age_LVEF_interaction'] = df_eng['age'] * (100 - df_eng['LVEF'])
        df_eng['age_adjusted_LVEF'] = df_eng['LVEF'] / (1 + (df_eng['age'] - 50) * 0.001)
        
        # 5. TREATMENT FEATURES
        df_eng['treatment_AC_only'] = ((df_eng['AC'] == 1) & (df_eng['antiHER2'] == 0)).astype(int)
        df_eng['treatment_antiHER2_only'] = ((df_eng['AC'] == 0) & (df_eng['antiHER2'] == 1)).astype(int)
        df_eng['treatment_combination'] = ((df_eng['AC'] == 1) & (df_eng['antiHER2'] == 1)).astype(int)
        df_eng['treatment_none'] = ((df_eng['AC'] == 0) & (df_eng['antiHER2'] == 0)).astype(int)
        df_eng['prior_treatment_count'] = df_eng['ACprev'].astype(int) + df_eng['RTprev'].astype(int)
        df_eng['current_treatment_count'] = df_eng['AC'].astype(int) + df_eng['antiHER2'].astype(int)
        df_eng['cumulative_cardiotoxic_treatment'] = (
            df_eng['AC'].astype(int) + df_eng['antiHER2'].astype(int) + 
            df_eng['ACprev'].astype(int) + df_eng['RTprev'].astype(int)
        )
        
        # 6. CARDIOVASCULAR RISK SCORE
        df_eng['n_risk_factors'] = (
            df_eng['HTA'].astype(int) + df_eng['DL'].astype(int) + 
            df_eng['smoker'].astype(int) + df_eng['diabetes'].astype(int) + 
            df_eng['obesity'].astype(int)
        )
        df_eng['CV_risk_score'] = (
            df_eng['age'] * 0.01 + 
            df_eng['n_risk_factors'] * 10 + 
            (100 - df_eng['LVEF']) * 0.5
        )
        
        # 7. INTERACTION FEATURES
        df_eng['age_x_AC'] = df_eng['age'] * df_eng['AC']
        df_eng['age_x_antiHER2'] = df_eng['age'] * df_eng['antiHER2']
        df_eng['LVEF_x_AC'] = df_eng['LVEF'] * df_eng['AC']
        df_eng['LVEF_x_antiHER2'] = df_eng['LVEF'] * df_eng['antiHER2']
        df_eng['LVEF_x_n_risk_factors'] = df_eng['LVEF'] * df_eng['n_risk_factors']
        df_eng['LVEF_x_HTA'] = df_eng['LVEF'] * df_eng['HTA']
        df_eng['BMI_x_combination'] = df_eng['BMI'] * df_eng['treatment_combination']
        df_eng['risk_x_treatment'] = df_eng['n_risk_factors'] * df_eng['cumulative_cardiotoxic_treatment']
        
        # 8. POLYNOMIAL FEATURES
        df_eng['age_squared'] = df_eng['age'] ** 2
        df_eng['age_cubed'] = df_eng['age'] ** 3
        df_eng['LVEF_squared'] = df_eng['LVEF'] ** 2
        df_eng['LVEF_cubed'] = df_eng['LVEF'] ** 3
        df_eng['heart_rate_squared'] = df_eng['heart_rate'] ** 2
        df_eng['heart_rate_cubed'] = df_eng['heart_rate'] ** 3
        
        # 9. RATIO FEATURES
        df_eng['LVSd_to_LVDd'] = df_eng['LVSd'] / df_eng['LVDd']
        df_eng['LAd_to_LVDd'] = df_eng['LAd'] / df_eng['LVDd']
        df_eng['PWT_to_LVDd'] = df_eng['PWT'] / df_eng['LVDd']
        df_eng['LVDd_to_BSA'] = df_eng['LVDd'] / df_eng['BSA']
        df_eng['weight_to_height'] = df_eng['weight'] / df_eng['height']
        
        return df_eng


class CardiotoxicityPredictor:
    """
    Complete predictor class that loads all components and provides prediction interface
    """
    
    def __init__(self, model_dir='.'):
        """
        Initialize predictor by loading all components
        
        Args:
            model_dir: Directory containing model files
        """
        model_dir = Path(model_dir)
        
        # Load model
        with open(model_dir / 'cardiotoxicity_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load feature statistics and create pipelines
        with open(model_dir / 'feature_statistics.json', 'r') as f:
            feature_stats = json.load(f)
        
        # Create preprocessing pipeline
        self.preprocessing = PreprocessingPipeline(feature_stats)
        
        # Create feature engineering pipeline
        self.feature_engineering = FeatureEngineeringPipeline()
        
        # Load metadata
        with open(model_dir / 'model_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.selected_features = self.metadata['features']['selected_features']
    
    def predict(self, patient_data):
        """
        Predict CTRCD risk for patient(s)
        
        Args:
            patient_data: DataFrame with patient information
            
        Returns:
            Dictionary with predictions and risk levels
        """
        # Validate
        is_valid, message = self.preprocessing.validate(patient_data)
        if not is_valid:
            raise ValueError(f"Validation error: {message}")
        
        # Preprocess
        processed = self.preprocessing.preprocess(patient_data)
        
        # Engineer features
        engineered = self.feature_engineering.engineer_features(processed)
        
        # Select features
        X = engineered[self.selected_features]
        
        # Predict
        probabilities = self.model.predict_proba(X)[:, 1]
        predictions = self.model.predict(X)
        
        # Interpret risk levels
        risk_levels = []
        recommendations = []
        
        for prob in probabilities:
            if prob < 0.3:
                risk_levels.append("Low")
                recommendations.append("Standard monitoring (every 6 months)")
            elif prob < 0.5:
                risk_levels.append("Moderate")
                recommendations.append("Enhanced monitoring (every 3 months)")
            elif prob < 0.7:
                risk_levels.append("High")
                recommendations.append("Cardioprotective agents + frequent monitoring")
            else:
                risk_levels.append("Very High")
                recommendations.append("Consider treatment modification + cardiology consult")
        
        return {
            'probabilities': probabilities.tolist(),
            'predictions': predictions.tolist(),
            'risk_levels': risk_levels,
            'recommendations': recommendations
        }
    
    def predict_single(self, patient_dict):
        """
        Predict for a single patient from dictionary
        
        Args:
            patient_dict: Dictionary with patient information
            
        Returns:
            Dictionary with prediction results
        """
        patient_df = pd.DataFrame([patient_dict])
        results = self.predict(patient_df)
        
        return {
            'probability': results['probabilities'][0],
            'prediction': results['predictions'][0],
            'risk_level': results['risk_levels'][0],
            'recommendation': results['recommendations'][0]
        }
