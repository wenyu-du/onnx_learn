#!/bin/bash

# Create a virtual environment named 'onnx_env'
python3 -m venv onnx_env

# Activate the virtual environment
source onnx_env/bin/activate

# Upgrade pip
pip install --upgrade pip

# Install requirements
if [ -f "tutorial/requirements.txt" ]; then
    pip install -r tutorial/requirements.txt
else
    pip install torch onnx onnxruntime numpy matplotlib netron
fi

echo "Virtual environment 'onnx_env' created and dependencies installed."
echo "To activate it, run: source onnx_env/bin/activate"
