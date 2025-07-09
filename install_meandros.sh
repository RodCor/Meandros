#!/bin/bash

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print with color
print_message() {
    local color=$1
    local message=$2
    echo -e "${color}${message}${NC}"
}

# Clean up __pycache__ directories
print_message "$YELLOW" "Cleaning up Python cache files..."
find . -type d -name "__pycache__" -exec rm -r {} +
find . -type f -name "*.pyc" -delete
find . -type f -name "*.pyo" -delete

# Create roi_detection_model directory if it doesn't exist
print_message "$YELLOW" "Setting up model directory..."

# Download model file if it doesn't exist
MODEL_FILE="roi_detection_module/mask_rcnn_axol_hand_ax.h5"
if [ ! -f "$MODEL_FILE" ]; then
    print_message "$YELLOW" "Downloading model file from Zenodo..."
    curl -L "https://zenodo.org/records/15036318/files/mask_rcnn_axol_hand_ax.h5?download=1" -o "$MODEL_FILE"
    if [ $? -ne 0 ]; then
        print_message "$RED" "Failed to download model file."
        exit 1
    fi
    print_message "$GREEN" "Model file downloaded successfully."
else
    print_message "$GREEN" "Model file already exists."
fi

# Check for Python installation
print_message "$YELLOW" "Checking Python installation..."
if ! command -v python3 >/dev/null 2>&1; then
    if ! command -v python >/dev/null 2>&1; then
        print_message "$RED" "Python is not installed. Please install Python 3.8 to 3.11."
        exit 1
    fi
    PYTHON_CMD="python"
else
    PYTHON_CMD="python3"
fi

# Check Python version without using bc
python_version=$($PYTHON_CMD -c 'import sys; v=sys.version_info; print(f"{v.major}.{v.minor}")')
python_major=$(echo $python_version | cut -d. -f1)
python_minor=$(echo $python_version | cut -d. -f2)

# Check if Python version is too low
if [ "$python_major" -lt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -lt 8 ]); then
    print_message "$RED" "Python version must be 3.8 or higher. Current version: $python_version"
    exit 1
fi

# Check if Python version is too high
if [ "$python_major" -gt 3 ] || ([ "$python_major" -eq 3 ] && [ "$python_minor" -gt 11 ]); then
    print_message "$RED" "Python version must be 3.11 or lower. Current version: $python_version"
    print_message "$RED" "Meandros requires Python 3.8 to 3.11."
    exit 1
fi

print_message "$GREEN" "Python $python_version found."

# Determine the virtual environment paths based on OS
if [[ "$OSTYPE" == "msys" ]] || [[ "$OSTYPE" == "win32" ]]; then
    # Windows paths (Git Bash or Command Prompt)
    VENV_PYTHON=".venv/Scripts/python.exe"
    VENV_PIP=".venv/Scripts/pip.exe"
else
    # Unix paths
    VENV_PYTHON=".venv/bin/python"
    VENV_PIP=".venv/bin/pip"
fi

# Remove existing virtual environment if it exists
if [ -d ".venv" ]; then
    print_message "$YELLOW" "Removing existing virtual environment..."
    rm -rf .venv
fi

# Create and set up virtual environment
print_message "$YELLOW" "Creating virtual environment..."
$PYTHON_CMD -m venv .venv --upgrade-deps
if [ $? -ne 0 ]; then
    print_message "$RED" "Failed to create virtual environment."
    exit 1
fi

# Ensure pip is installed in virtual environment
print_message "$YELLOW" "Ensuring pip is available..."
if [ ! -f "$VENV_PIP" ]; then
    print_message "$YELLOW" "Installing pip in virtual environment..."
    curl -sSL https://bootstrap.pypa.io/get-pip.py -o get-pip.py
    $VENV_PYTHON get-pip.py --force-reinstall
    rm get-pip.py
fi

# Upgrade pip in virtual environment
print_message "$YELLOW" "Upgrading pip in virtual environment..."
$VENV_PYTHON -m pip install --upgrade pip

# Install UV in virtual environment
print_message "$YELLOW" "Installing UV in virtual environment..."
$VENV_PYTHON -m pip install --upgrade uv

# Install dependencies using UV from virtual environment
print_message "$YELLOW" "Installing Meandros dependencies..."

# Debug: Show which Python UV is detecting
print_message "$YELLOW" "Checking Python version UV will use..."
$VENV_PYTHON -c "import sys; print(f'Virtual environment Python: {sys.version}')"

if [ -f "requirements.lock" ]; then
    print_message "$YELLOW" "Using requirements.lock for exact versions..."
    # Use --python flag to explicitly specify the Python interpreter
    $VENV_PYTHON -m uv pip sync requirements.lock --python "$VENV_PYTHON"
else
    print_message "$YELLOW" "Installing from pyproject.toml..."
    # Use --python flag to explicitly specify the Python interpreter
    $VENV_PYTHON -m uv pip install . --python "$VENV_PYTHON"
fi

if [ $? -eq 0 ]; then
    print_message "$GREEN" "Meandros installation completed successfully!"
    print_message "$YELLOW" "To run Meandros, use: ./run_meandros.sh"
else
    print_message "$RED" "Installation failed. Please check the error messages above."
    exit 1
fi 