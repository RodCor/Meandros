#!/bin/bash

# Check if .venv exists
if [ ! -d ".venv" ]; then
    echo "Virtual environment (.venv) not found. Please run 'install_meandros.sh' first to set up the environment."
    exit 1
fi

# Activate virtual environment and run meandros
echo "Starting Meandros..."
if [ -f ".venv/Scripts/activate" ]; then
    # Windows path (Git Bash or WSL)
    source .venv/Scripts/activate
else
    # Unix path
    source .venv/bin/activate
fi

# Run meandros.py from within the activated environment
python meandros.py

# Deactivate virtual environment
deactivate 