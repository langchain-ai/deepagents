#!/bin/bash
# LXPLUS setup script for Chatlas Agents environment
setupATLAS

# Set up the UV cache directory in tmp
export UV_CACHE_DIR="/tmp/${USER}/uv-cache-asw"
mkdir -p ${UV_CACHE_DIR}

# Optionally set up Rucio and AMI, if running on CVMFS ATLAS enabled system
# Allows agent to call these commands through deepagents SKILLS interface

# AMI skill for grid samples metadata 
OPT_SETUP_AMI="true"

# Rucio skill for additional grid info (To be implemented)
# OPT_SETUP_RUCIO="true"
OPT_SETUP_RUCIO="false"

if [ -e .atlas-pip/setup.sh ]; then
    echo "Pip already set up with SetupATLAS, skipping pip setup."
else
    echo "Setting up .atlas-pip environment..."
    installPip -p 3.13 .atlas-pip/requirements.txt 
fi

# Initialize VOMS proxy if either setup option is enabled
if [ "${OPT_SETUP_AMI}" = "true" ] || [ "${OPT_SETUP_RUCIO}" = "true" ]; then
    echo "Initializing VOMS proxy for ATLAS..."
    voms-proxy-init --voms atlas
    if [ "${OPT_SETUP_AMI}" = "true" ]; then
        echo "Setting up AMI..."
        lsetup pyami
        cp -r skills/ami-query ../.deepagents/skills/
    fi
    if [ "${OPT_SETUP_RUCIO}" = "true" ]; then
        echo "Setting up Rucio..."
        lsetup rucio
        cp -r skills/rucio-management ../.deepagents/skills/
    fi
fi

source .atlas-pip/setup.sh
source .env
uv sync 