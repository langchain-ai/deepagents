#!/bin/bash
# LXPLUS setup script for Chatlas Agents environment
setupATLAS

if [ -e .atlas-pip/setup.sh ]; then
    echo "Pip already set up with SetupATLAS, skipping pip setup."
else
    echo "Setting up .atlas-pip environment..."
    installPip -p 3.13 .atlas-pip/requirements.txt 
fi

source .atlas-pip/setup.sh
source .env
uv sync 