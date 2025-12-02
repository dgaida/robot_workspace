# Installation Guide

Complete installation instructions for the Robot Workspace package.

## Table of Contents

- [Prerequisites](#prerequisites)
- [Installation Methods](#installation-methods)
  - [Standard Installation](#standard-installation)
  - [Development Installation](#development-installation)
  - [Robot-Specific Installation](#robot-specific-installation)
- [Verification](#verification)
- [Platform-Specific Notes](#platform-specific-notes)
- [Troubleshooting](#troubleshooting)

## Prerequisites

### System Requirements

- **Python**: 3.9 or higher (3.11 recommended)
- **Operating System**: Linux, Windows, or macOS
- **Memory**: Minimum 2GB RAM
- **Disk Space**: ~500MB for package and dependencies

### Required System Packages

#### Ubuntu/Debian
```bash
sudo apt-get update
sudo apt-get install -y python3-pip python3-dev build-essential
sudo apt-get install -y libopencv-dev python3-opencv
```

#### macOS
```bash
brew install python@3.11
brew install opencv
```

#### Windows
- Install Python from [python.org](https://www.python.org/downloads/)
- Visual Studio Build Tools may be required for some dependencies

## Installation Methods

### Standard Installation

For general use without robot hardware:

```bash
# Clone the repository
git clone https://github.com/dgaida/robot_workspace.git
cd robot_workspace

# Install in standard mode
pip install -e .
```

This installs:
- Core workspace management
- Object detection and representation
- Coordinate transformations
- Basic utilities

### Development Installation

For contributors and developers:

```bash
# Clone the repository
git clone https://github.com/dgaida/robot_workspace.git
cd robot_workspace

# Install with development dependencies
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install
```

This includes:
- All standard features
- Testing tools (pytest, pytest-cov)
- Code formatting (black, ruff)
- Type checking (mypy)
- Pre-commit hooks

### Robot-Specific Installation

#### For Niryo Ned2 Robot

```bash
# Install with Niryo support
pip install -e ".[niryo]"
```

Additional requirements:
- Access to Niryo Ned2 robot (real or simulated)
- Network connectivity to robot
- PyNiryo library (installed automatically)

#### For WidowX Robot

```bash
# Install standard version (WidowX support is built-in)
pip install -e .
```

Additional requirements:
- Access to WidowX 250 6DOF robot
- ROS installation (if using ROS interface)
- Third-person camera setup (e.g., Intel RealSense)

### Complete Installation (All Features)

```bash
# Install everything
pip install -e ".[all]"
```

## Verification

### Quick Test

```bash
# Run the demo script
python main.py
```

Expected output:
- Pose object demonstrations
- Workspace management examples
- Object creation and manipulation
- Serialization examples
- LLM formatting demonstrations

### Run Test Suite

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=robot_workspace --cov-report=term-missing

# Run specific test categories
pytest tests/objects/      # Object tests only
pytest tests/workspaces/   # Workspace tests only
```

### Verify Installation

```python
# In Python interpreter
import robot_workspace
print(robot_workspace.__version__)  # Should print version number

# Test basic functionality
from robot_workspace import PoseObjectPNP, Location
pose = PoseObjectPNP(0.2, 0.1, 0.3, 0.0, 1.57, 0.0)
print(pose)  # Should display pose information
```

## Platform-Specific Notes

### Linux (Ubuntu/Debian)

Recommended for production use. Best compatibility with robot hardware.

**Additional setup for camera access:**
```bash
# Add user to video group
sudo usermod -a -G video $USER
# Logout and login for changes to take effect
```

### macOS

Fully supported for development and testing.

**Known issues:**
- Some USB camera drivers may require additional setup

### Windows

Supported with some limitations.

**Important notes:**
- Use Command Prompt or PowerShell (not Git Bash for installation)
- May need Visual Studio Build Tools for OpenCV compilation

**Visual Studio Build Tools:**
Download from [Microsoft](https://visualstudio.microsoft.com/downloads/) and install "Desktop development with C++"

## Virtual Environment (Recommended)

### Using venv

```bash
# Create virtual environment
python -m venv robot_env

# Activate (Linux/macOS)
source robot_env/bin/activate

# Activate (Windows)
robot_env\Scripts\activate

# Install package
pip install -e .
```

### Using conda

```bash
# Create conda environment
conda create -n robot_workspace python=3.11

# Activate environment
conda activate robot_workspace

# Install package
pip install -e .
```

### Using the provided environment.yaml

```bash
# Create environment from file
conda env create -f environment.yaml

# Activate environment
conda activate llm_niryo_cuda12

# Install package
pip install -e .
```

## Troubleshooting

### Common Issues

#### ImportError: No module named 'cv2'

**Solution:**
```bash
pip install opencv-python
# or
sudo apt-get install python3-opencv
```

#### ImportError: No module named 'pyniryo'

**Solution:**
```bash
pip install pyniryo
# or install with Niryo support
pip install -e ".[niryo]"
```

#### Permission denied when accessing robot

**Solution (Linux):**
```bash
sudo usermod -a -G dialout $USER
# Logout and login
```

#### Tests fail with "No module named robot_workspace"

**Solution:**
```bash
# Ensure you installed in editable mode
pip install -e .

# Or add to PYTHONPATH
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
```

#### OpenCV errors on import

**Solution:**
```bash
# Uninstall all OpenCV versions
pip uninstall opencv-python opencv-python-headless opencv-contrib-python

# Reinstall the correct one
pip install opencv-python
```

### Getting Help

If you encounter issues not covered here:

1. **Check existing issues**: [GitHub Issues](https://github.com/dgaida/robot_workspace/issues)
2. **Create new issue**: Provide:
   - Python version: `python --version`
   - Operating system
   - Installation method used
   - Full error message
   - Steps to reproduce

## Next Steps

After successful installation:

1. **Read the documentation**: See [docs/README.md](README.md)
2. **Run examples**: Execute `python main.py`
3. **Explore the API**: Review [docs/api.md](api.md)
4. **Check examples**: See [docs/examples.md](examples.md)
5. **Run tests**: Execute `pytest`

## Development Setup

For contributors:

```bash
# Fork and clone
git clone https://github.com/YOUR_USERNAME/robot_workspace.git
cd robot_workspace

# Install development version
pip install -e ".[dev]"

# Install pre-commit hooks
pre-commit install

# Create feature branch
git checkout -b feature/my-feature

# Make changes and test
pytest

# Format code
black .
ruff check . --fix

# Commit and push
git commit -am "Add feature"
git push origin feature/my-feature
```

## Uninstallation

To remove the package:

```bash
# Uninstall package
pip uninstall robot-workspace

# Remove virtual environment (if used)
rm -rf robot_env  # or your venv name

# Remove conda environment (if used)
conda env remove -n robot_workspace
```

## Updates

To update to the latest version:

```bash
# If installed from source
cd robot_workspace
git pull origin master
pip install -e . --upgrade

# If installed from PyPI (when available)
pip install --upgrade robot-workspace
```

---

**Author**: Daniel Gaida
**Email**: daniel.gaida@th-koeln.de
**Repository**: https://github.com/dgaida/robot_workspace
