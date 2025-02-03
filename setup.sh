#!/bin/bash

# List contents of /users/swargo98/.local/bin/
ls /users/swargo98/.local/bin/

# Add directory to PATH
export PATH=$PATH:/users/swargo98/.local/bin

# Persist the PATH change in ~/.bashrc
echo 'export PATH=$PATH:/users/swargo98/.local/bin' >> ~/.bashrc

# Reload ~/.bashrc
source ~/.bashrc

# Install virtualenv
pip install virtualenv

# Create a virtual environment
virtualenv venv

# Activate the virtual environment
source venv/bin/activate

echo "Environment setup is complete. Virtual environment 'venv' is activated."