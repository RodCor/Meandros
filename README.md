# <img src="meandros_logo.ico" width="32" height="32" alt="Meandros Logo"> Meandros

## Overview

Meandros is a specialized tool for analyzing and processing biological images, with a focus on axolotl limb development and regeneration studies. The software provides a comprehensive suite of tools for ROI detection, axis approximation, and statistical analysis.

## System Requirements

- Python 3.8 or higher
- Git (for cloning the repository)
- Operating Systems supported:
  - Windows (with Git Bash or WSL)
  - Linux
  - macOS

## Installation

### Quick Start

1. Get the code (choose one option):
   
   a. Clone from GitHub:
   ```bash
   git clone https://github.com/RodCor/Meandros
   cd Meandros
   ```

   b. Download from Zenodo:
   - Download and extract the ZIP file
   - Navigate to the extracted folder

2. Run the installation script:
   ```bash
   # On Windows (using Git Bash)
   bash install_meandros.sh

   # On Linux/macOS
   ./install_meandros.sh
   ```

The installation script will:
- Check for Python installation
- Create a virtual environment
- Install all required dependencies
- Set up the necessary configuration

### Manual Installation (Alternative)

If you prefer to install manually, you can:

1. Create a virtual environment:
   ```bash
   python -m venv .venv
   ```

2. Activate the virtual environment:
   ```bash
   # On Windows
   .venv\Scripts\activate

   # On Linux/macOS
   source .venv/bin/activate
   ```

3. Install dependencies:
   ```bash
   python -m pip install --upgrade pip
   python -m pip install uv
   python -m uv pip install .
   ```

## Running Meandros

After installation, you can run Meandros using:

```bash
# On Windows (using Git Bash)
bash run_meandros.sh

# On Linux/macOS
./run_meandros.sh
```

## Features

- ROI (Region of Interest) Detection
- Axis Approximation
- Statistical Analysis
- Data Export Capabilities
- Interactive Visualization

## Troubleshooting

If you encounter any issues during installation:

1. Ensure Python 3.8+ is installed:
   ```bash
   python --version
   ```

2. Check if the virtual environment was created properly:
   ```bash
   # The .venv directory should exist
   ls .venv
   ```

3. Try removing the virtual environment and running the installation script again:
   ```bash
   rm -rf .venv
   bash install_meandros.sh
   ```


## License

MIT License Copyright © 2025

## Contact

For questions and support, please contact the Authors of the project. 