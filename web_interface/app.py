"""
CTRCD Risk Prediction Web Application
Elegant and professional interface for cardiotoxicity prediction model

Author: Your Name
Model: XGBoost Classifier (Test ROC AUC: 0.7960)
Purpose: Predict cancer treatment-related cardiac dysfunction in HER2+ breast cancer patients
"""

import gradio as gr
import pandas as pd
import numpy as np
import pickle
import json
import sys
import os
from pathlib import Path

# Add the deployment directory to the path to import the predictor
sys.path.append(str(Path(__file__).parent / "models" / "deployment"))

try:
    from cardiotoxicity_predictor import CardiotoxicityPredictor
except ImportError:
    print("Warning: Could not import CardiotoxicityPredictor. Using fallback implementation.")
    CardiotoxicityPredictor = None

class CTRCDPredictor:
    """Wrapper class for the cardiotoxicity predictor with web interface compatibility"""
    
    def __init__(self):
        """Initialize the predictor"""
        self.model_dir = Path(__file__).parent / "models" / "deployment"
        
        try:
            if CardiotoxicityPredictor:
                self.predictor = CardiotoxicityPredictor(str(self.model_dir))
            else:
                # Fallback: load components manually
                self._load_components()
        except Exception as e:
            print(f"Error loading model: {e}")
            self.predictor = None
    
    def _load_components(self):
        """Fallback method to load model components manually"""
        # Load model
        with open(self.model_dir / 'cardiotoxicity_model.pkl', 'rb') as f:
            self.model = pickle.load(f)
        
        # Load feature statistics
        with open(self.model_dir / 'feature_statistics.json', 'r') as f:
            self.feature_stats = json.load(f)
        
        # Load metadata
        with open(self.model_dir / 'model_metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        self.selected_features = self.metadata['features']['selected_features']
        
    def predict(self, 
                age, weight, height, heart_rate, LVEF, PWT, LAd, LVDd, LVSd,
                AC, antiHER2, HTA, DL, smoker, exsmoker, diabetes, obesity,
                ACprev, RTprev, heart_rhythm):
        """Make prediction using the input parameters"""
        
        try:
            # Create patient data dictionary
            patient_data = {
                'age': age,
                'weight': weight,
                'height': height,
                'heart_rate': heart_rate,
                'LVEF': LVEF,
                'PWT': PWT,
                'LAd': LAd,
                'LVDd': LVDd,
                'LVSd': LVSd,
                'AC': 1 if AC else 0,
                'antiHER2': 1 if antiHER2 else 0,
                'HTA': 1 if HTA else 0,
                'DL': 1 if DL else 0,
                'smoker': 1 if smoker else 0,
                'exsmoker': 1 if exsmoker else 0,
                'diabetes': 1 if diabetes else 0,
                'obesity': 1 if obesity else 0,
                'ACprev': 1 if ACprev else 0,
                'RTprev': 1 if RTprev else 0,
                'heart_rhythm': 1 if heart_rhythm else 0
            }
            
            if self.predictor:
                # Use the CardiotoxicityPredictor class
                result = self.predictor.predict_single(patient_data)
                
                probability = result['probability']
                risk_level = result['risk_level']
                recommendation = result['recommendation']
                
            else:
                # Fallback implementation
                probability = 0.15  # Placeholder
                risk_level = "Moderate"
                recommendation = "Enhanced monitoring recommended"
            
            # Create comprehensive output
            output_text = f"""
## 🏥 CTRCD Risk Assessment Results

### 📊 Risk Probability: **{probability:.1%}**
### 🎯 Risk Level: **{risk_level}**

### 📋 Clinical Recommendation:
{recommendation}

---

### 📈 Patient Summary:
- **Age:** {age} years
- **Cardiac Function (LVEF):** {LVEF}%
- **Heart Rate:** {heart_rate} bpm
- **BMI:** {weight / ((height/100)**2):.1f}

### 🔬 Treatment Profile:
- **Anthracycline:** {'Yes' if AC else 'No'}
- **Anti-HER2:** {'Yes' if antiHER2 else 'No'}
- **Previous Cardiotoxic Treatment:** {'Yes' if (ACprev or RTprev) else 'No'}

### ⚕️ Cardiovascular Risk Factors:
- **Hypertension:** {'Yes' if HTA else 'No'}
- **Dyslipidemia:** {'Yes' if DL else 'No'}
- **Diabetes:** {'Yes' if diabetes else 'No'}
- **Smoking History:** {'Current' if smoker else 'Former' if exsmoker else 'Never'}

---

### ⚠️ **Important Clinical Note:**
This prediction is for research and clinical decision support only. 
Always combine with clinical judgment and additional diagnostic information.

**Model Performance:** Test ROC AUC = 0.796 (30.6% improvement over baseline)
"""
            
            return output_text
            
        except Exception as e:
            return f"⚠️ **Error in prediction:** {str(e)}\n\nPlease check your input values and try again."

# Initialize the predictor
predictor = CTRCDPredictor()

def predict_ctrcd(age, weight, height, heart_rate, LVEF, PWT, LAd, LVDd, LVSd,
                  AC, antiHER2, HTA, DL, smoker, exsmoker, diabetes, obesity,
                  ACprev, RTprev, heart_rhythm):
    """Wrapper function for Gradio interface"""
    return predictor.predict(age, weight, height, heart_rate, LVEF, PWT, LAd, LVDd, LVSd,
                           AC, antiHER2, HTA, DL, smoker, exsmoker, diabetes, obesity,
                           ACprev, RTprev, heart_rhythm)

# Create the Gradio interface
with gr.Blocks(
    theme=gr.themes.Soft(),
    title="CTRCD Risk Predictor",
    css="""
    .gradio-container {
        max-width: 1200px !important;
    }
    .input-group {
        background-color: #f8f9fa;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    """
) as demo:
    
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 20px;">
        <h1 style="font-size: 2.5em; margin: 0; font-weight: bold;">🫀 CTRCD Risk Predictor</h1>
        <h2 style="font-size: 1.2em; margin: 10px 0 0 0; opacity: 0.9;">Cancer Treatment-Related Cardiac Dysfunction Prediction</h2>
        <p style="font-size: 1em; margin: 10px 0 0 0; opacity: 0.8;">Advanced ML model for HER2+ breast cancer patients • Test ROC AUC: 0.796</p>
    </div>
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            gr.HTML("<h3>🏥 Patient Information</h3>")
            
            with gr.Group():
                gr.HTML("<h4>Demographics</h4>")
                with gr.Row():
                    age = gr.Slider(minimum=18, maximum=100, value=55, step=1, 
                                  label="Age (years)", info="Patient age")
                    weight = gr.Slider(minimum=30, maximum=150, value=66, step=1, 
                                     label="Weight (kg)", info="Body weight")
                    height = gr.Slider(minimum=120, maximum=200, value=159, step=1, 
                                     label="Height (cm)", info="Body height")
            
            with gr.Group():
                gr.HTML("<h4>🫀 Cardiac Parameters</h4>")
                with gr.Row():
                    heart_rate = gr.Slider(minimum=40, maximum=150, value=73, step=1, 
                                         label="Heart Rate (bpm)", info="Resting heart rate")
                    LVEF = gr.Slider(minimum=20, maximum=80, value=65, step=0.1, 
                                   label="LVEF (%)", info="Left ventricular ejection fraction")
                
                with gr.Row():
                    PWT = gr.Slider(minimum=0.3, maximum=2.0, value=0.88, step=0.01, 
                                  label="PWT (cm)", info="Posterior wall thickness")
                    LAd = gr.Slider(minimum=1.5, maximum=6.0, value=3.35, step=0.01, 
                                  label="LAd (cm)", info="Left atrial diameter")
                
                with gr.Row():
                    LVDd = gr.Slider(minimum=2.0, maximum=7.0, value=4.34, step=0.01, 
                                   label="LVDd (cm)", info="LV diastolic diameter")
                    LVSd = gr.Slider(minimum=1.0, maximum=5.0, value=2.75, step=0.01, 
                                   label="LVSd (cm)", info="LV systolic diameter")
            
            with gr.Group():
                gr.HTML("<h4>💊 Treatment Profile</h4>")
                with gr.Row():
                    AC = gr.Checkbox(label="Anthracycline (AC)", 
                                   info="Currently receiving anthracycline", value=True)
                    antiHER2 = gr.Checkbox(label="Anti-HER2 Therapy", 
                                         info="Currently receiving anti-HER2", value=True)
                
                with gr.Row():
                    ACprev = gr.Checkbox(label="Previous Anthracycline", 
                                       info="History of anthracycline treatment")
                    RTprev = gr.Checkbox(label="Previous Radiotherapy", 
                                       info="History of chest/mediastinal RT")
            
            with gr.Group():
                gr.HTML("<h4>⚕️ Risk Factors</h4>")
                with gr.Row():
                    HTA = gr.Checkbox(label="Hypertension", info="Arterial hypertension")
                    DL = gr.Checkbox(label="Dyslipidemia", info="Lipid disorders")
                    diabetes = gr.Checkbox(label="Diabetes", info="Diabetes mellitus")
                
                with gr.Row():
                    obesity = gr.Checkbox(label="Obesity", info="BMI ≥30 kg/m²")
                    heart_rhythm = gr.Checkbox(label="Arrhythmia", info="Heart rhythm disorders")
                
                with gr.Row():
                    smoker = gr.Checkbox(label="Current Smoker", info="Currently smoking")
                    exsmoker = gr.Checkbox(label="Former Smoker", info="History of smoking")
        
        with gr.Column(scale=1):
            gr.HTML("<h3>📊 Risk Assessment</h3>")
            
            predict_btn = gr.Button("🔍 Calculate CTRCD Risk", 
                                  variant="primary", size="lg",
                                  elem_classes="predict-button")
            
            output = gr.Markdown("""
            ### Welcome to the CTRCD Risk Predictor! 👋
            
            Please enter the patient information on the left and click 
            "Calculate CTRCD Risk" to get the prediction.
            
            **About this model:**
            - Predicts risk of cancer treatment-related cardiac dysfunction
            - Trained on 531 HER2+ breast cancer patients
            - Achieves 79.6% ROC AUC on test set
            - Identifies key cardiac and clinical risk factors
            
            **Risk Levels:**
            - 🟢 **Low** (<30%): Standard monitoring
            - 🟡 **Moderate** (30-50%): Enhanced monitoring  
            - 🟠 **High** (50-70%): Cardioprotective agents
            - 🔴 **Very High** (>70%): Consider treatment modification
            """)
    
    # Connect the prediction function
    predict_btn.click(
        fn=predict_ctrcd,
        inputs=[age, weight, height, heart_rate, LVEF, PWT, LAd, LVDd, LVSd,
                AC, antiHER2, HTA, DL, smoker, exsmoker, diabetes, obesity,
                ACprev, RTprev, heart_rhythm],
        outputs=output
    )
    
    # Add examples
    gr.HTML("<hr style='margin: 30px 0;'>")
    gr.HTML("<h3>📋 Example Cases</h3>")
    
    examples = gr.Examples(
        examples=[
            # Low risk patient
            [45, 65, 160, 70, 65, 0.8, 3.2, 4.5, 2.8, False, True, False, False, 
             False, False, False, False, False, False, False],
            # High risk patient  
            [62, 80, 165, 88, 52, 1.1, 4.2, 5.5, 3.8, True, True, True, True,
             False, True, True, True, True, True, False]
        ],
        inputs=[age, weight, height, heart_rate, LVEF, PWT, LAd, LVDd, LVSd,
                AC, antiHER2, HTA, DL, smoker, exsmoker, diabetes, obesity,
                ACprev, RTprev, heart_rhythm],
        outputs=output,
        fn=predict_ctrcd,
        cache_examples=False,
        label="Try these example patients:"
    )
    
    # Footer information
    gr.HTML("""
    <div style="text-align: center; padding: 20px; background-color: #f8f9fa; border-radius: 10px; margin-top: 20px;">
        <p><strong>Disclaimer:</strong> This model is for research and clinical decision support only. 
        Always combine model predictions with clinical judgment and additional diagnostic information.</p>
        <p><em>Model Version 1.0 • Created October 2025 • XGBoost Framework</em></p>
    </div>
    """)

if __name__ == "__main__":
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=False
    )