#!/bin/bash

# CTRCD Risk Predictor - Local Development Startup Script
# This script sets up and runs the web application locally

echo "🫀 CTRCD Risk Predictor - Startup Script"
echo "========================================"

# Check if virtual environment exists
if [ ! -d "venv" ]; then
    echo "Creating virtual environment..."
    python -m venv venv
fi

# Activate virtual environment and install dependencies
echo "Activating virtual environment and installing dependencies..."
source venv/bin/activate
pip install -r requirements.txt

echo ""
echo "🚀 Starting the CTRCD Risk Predictor web application..."
echo "The application will be available at: http://localhost:7860"
echo "Press Ctrl+C to stop the application"
echo ""

# Run the application
python app.py