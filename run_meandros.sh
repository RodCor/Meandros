#!/bin/bash

# Function to check if a command exists
command_exists() {
    command -v "$1" >/dev/null 2>&1
}

# Check if UV is installed
if ! command_exists uv; then
    echo "UV is not installed. Please install it with: pip install uv"
    exit 1
fi

# Check if .venv exists, if not create it
if [ ! -d ".venv" ]; then
    echo "Creating virtual environment..."
    uv venv
    if [ $? -ne 0 ]; then
        echo "Failed to create virtual environment."
        exit 1
    fi
    
    echo "Installing dependencies..."
    if [ -f ".venv/Scripts/python.exe" ]; then
        # Windows path (Git Bash or WSL)
        ./.venv/Scripts/python.exe -m uv pip install .
    else
        # Unix path
        ./.venv/bin/python -m uv pip install .
    fi
    
    if [ $? -ne 0 ]; then
        echo "Failed to install dependencies."
        exit 1
    fi
fi

# Run meandros.py with the virtual environment's Python
echo "Starting Meandros..."
if [ -f ".venv/Scripts/python.exe" ]; then
    # Windows path (Git Bash or WSL)
    ./.venv/Scripts/python.exe meandros.py
else
    # Unix path
    ./.venv/bin/python meandros.py
fi 