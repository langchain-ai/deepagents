#!/bin/bash
# LXPLUS setup script for Chatlas Agents environment
setupATLAS

# Set up the UV cache directory in tmp
export UV_CACHE_DIR="/tmp/${USER}/uv-cache-asw"
mkdir -p $UV_CACHE_DIR

if [ -e .atlas-pip/setup.sh ]; then
    echo "Pip already set up with SetupATLAS, skipping pip setup."
else
    echo "Setting up .atlas-pip environment..."
    installPip -p 3.13 .atlas-pip/requirements.txt 
fi

source .atlas-pip/setup.sh
source .env
uv sync 