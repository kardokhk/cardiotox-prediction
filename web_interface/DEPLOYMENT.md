# 🫀 CTRCD Risk Predictor - Deployment Guide

## 🎯 Overview

This guide helps you deploy the CTRCD Risk Predictor web application either locally or on Hugging Face Spaces.

## 📦 Package Contents

- `app.py` - Main Gradio application
- `models/` - Complete model deployment package (500+ KB)
- `requirements.txt` - Python dependencies
- `README.md` - Comprehensive documentation
- `README_HF.md` - Hugging Face Spaces README
- `run_local.sh` - Local development script
- `venv/` - Virtual environment (local only)

## 🚀 Deployment Options

### Option 1: Hugging Face Spaces (Recommended)

1. **Create a New Space**
   - Go to [Hugging Face Spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose **Gradio** as SDK
   - Set visibility as desired

2. **Upload Files**
   ```bash
   # Copy these files to your HF Space:
   app.py
   requirements.txt
   models/ (entire directory)
   ```

3. **Set README**
   - Copy content from `README_HF.md` to your Space's README
   - The app will automatically deploy!

4. **Access Your App**
   - Your app will be available at: `https://huggingface.co/spaces/YOUR_USERNAME/YOUR_SPACE_NAME`

### Option 2: Local Development

1. **Quick Start**
   ```bash
   chmod +x run_local.sh
   ./run_local.sh
   ```

2. **Manual Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Run application
   python app.py
   ```

3. **Access Locally**
   - Open: `http://localhost:7860`

## 🔧 Configuration

### Environment Variables (Optional)
- `GRADIO_SERVER_NAME` - Server host (default: "0.0.0.0")
- `GRADIO_SERVER_PORT` - Server port (default: 7860)

### Customization
- Modify `app.py` for interface changes
- Update model files in `models/deployment/` for new models
- Adjust `requirements.txt` for dependency changes

## 📊 Model Information

- **Framework**: XGBoost 2.1.1
- **Performance**: Test ROC AUC = 0.796
- **Size**: ~500 KB total deployment package
- **Features**: 40 clinical parameters
- **Training**: 531 HER2+ breast cancer patients

## 🛠️ Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install setuptools  # Fixes distutils issues
   ```

2. **XGBoost Warnings**
   - These are informational only and don't affect functionality
   - Model still provides accurate predictions

3. **Memory Issues**
   - Ensure at least 1GB RAM available
   - Model loads ~300MB into memory

4. **Port Conflicts**
   - Change port in `app.py`: `demo.launch(server_port=8080)`

### Verification

Test the model is working:
```python
# Test prediction
from models.deployment.cardiotoxicity_predictor import CardiotoxicityPredictor
predictor = CardiotoxicityPredictor('models/deployment')

# Should return risk assessment without errors
result = predictor.predict_single({
    'age': 55, 'weight': 70, 'height': 165, 'heart_rate': 75,
    'LVEF': 60, 'PWT': 0.9, 'LAd': 3.5, 'LVDd': 4.8, 'LVSd': 3.2,
    'AC': 1, 'antiHER2': 1, 'HTA': 0, 'DL': 0, 'smoker': 0,
    'exsmoker': 0, 'diabetes': 0, 'obesity': 0, 'ACprev': 0,
    'RTprev': 0, 'heart_rhythm': 0
})
```

## 📚 Additional Resources

- **Model Documentation**: `models/deployment/README.md`
- **Usage Instructions**: `models/deployment/USAGE_INSTRUCTIONS.md`
- **Model Metadata**: `models/deployment/model_metadata.json`
- **Feature Statistics**: `models/deployment/feature_statistics.json`

## ⚠️ Important Notes

- **Clinical Use**: For research and clinical decision support only
- **Validation**: Tested on HER2+ breast cancer patients
- **Integration**: Always combine with clinical judgment
- **Updates**: Monitor for model version updates

## 🆘 Support

For technical issues:
1. Check model files are present and accessible
2. Verify all dependencies are installed
3. Review error logs for specific issues
4. Test with provided example cases

---

**Version**: 1.0  
**Last Updated**: October 2025  
**Model Performance**: Test ROC AUC = 0.796