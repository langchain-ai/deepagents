---
name: atlas-runquery
description: Query ATLAS run and data quality information for understanding detector conditions, data-taking periods, and luminosity on LXPlus
---

# ATLAS Run Query Skill

This skill provides guidance for querying ATLAS run information, data quality, and luminosity records to understand detector conditions and select appropriate data for physics analysis.

## Description

Run queries in ATLAS help you find information about data-taking runs, including detector status, data quality flags, luminosity, and trigger configurations. This is essential for selecting good runs for physics analysis and understanding detector conditions during data collection.

## When to Use This Skill

Use this skill when you need to:
- Find runs from specific data-taking periods
- Check data quality flags for analysis
- Query luminosity information for runs
- Identify runs with specific trigger configurations
- Check detector subsystem status during runs
- Create Good Runs Lists (GRLs) for analysis
- Understand beam conditions and pile-up
- Investigate run-level issues or anomalies

## Prerequisites

This skill assumes you are running on the CERN LXPlus cluster with:
- Valid ATLAS computing account
- Access to ATLAS software stack via CVMFS
- **ATLAS environment already initialized** (user must run `setupATLAS` before launching the agent)
- **pyAMI tools already set up** (user must run `lsetup pyami` before launching the agent if using AMI for run queries)
- **Active VOMS proxy** (user must run `voms-proxy-init -voms atlas` before launching the agent)
- No additional software installation needed

**Important**: The user must set up the ATLAS environment and VOMS proxy **before** starting the agent session. If AMI is used for run queries, pyAMI must also be set up. If these are not set up, queries will fail with authentication errors.

### Verifying LXPlus Environment

To verify you are running on an LXPlus machine, check the hostname:
```bash
echo $HOSTNAME
```

LXPlus hostnames typically follow the pattern `lxplus*.cern.ch` (e.g., `lxplus7.cern.ch`, `lxplus8.cern.ch`). If the hostname does not contain "lxplus" and "cern.ch", you may not be on an LXPlus machine, and ATLAS tools may not be available.

## Important Note About Web-Based Tools

**The agent operates in a command-line environment only.** While ATLAS provides web interfaces for Run Query (atlas-runquery.cern.ch) and documentation (Twiki pages), these are **not accessible** to the agent.

**Agent Limitations:**
- Cannot access ATLAS internal websites (Run Query web, Twiki, etc.)
- Cannot use web browsers or make HTTP requests to ATLAS-internal URLs
- Cannot download GRL XML files from web URLs
- Must rely on command-line tools (AMI) or user-provided files

**For run queries, the agent can:**
- Use AMI command-line tool to query run information: `ami show run <run_number>`
- Work with GRL XML files that the user has already downloaded to the local filesystem
- Users must obtain GRLs from official sources and provide file paths to the agent

## Available Tools for Run Queries

### AMI (ATLAS Metadata Interface)

AMI provides run-level information via command line:
```bash
# Query runs in AMI (user must have already run lsetup pyami before starting agent)
ami show run <run_number>
```

**Note**: If you see authentication errors, the user's VOMS proxy may have expired. The user must run `voms-proxy-init -voms atlas` in their shell before restarting the agent.

### Good Runs Lists (GRL)

GRLs are pre-computed lists of runs passing data quality requirements:
- Published by ATLAS Data Preparation group
- Available in XML format
- Used in analysis frameworks (AnalysisTop, AthAnalysis)
- **Users must download GRL files** and provide local file paths to the agent

## How to Use

### Step 1: Query Run Information via AMI

Use AMI command-line tool to query specific runs:

```bash
# Get run information (user must have already run setupATLAS, lsetup pyami, and voms-proxy-init)
ami show run <run_number>
```

This provides information about:
- Detector status (subsystems on/off)
- Trigger configuration
- Luminosity blocks recorded
- Data streams available

### Step 2: Working with Good Runs Lists

For physics analysis, you need runs with good data quality through GRL XML files.

**Using Official GRLs:**

Users must obtain official GRL files from ATLAS Data Preparation group (agent cannot access web):
1. User downloads appropriate GRL XML file for their analysis from official ATLAS sources
2. User provides the local file path to the agent
3. Agent can read and use the GRL file in analysis workflows

**Example usage with local GRL file:**
```bash
# User has downloaded GRL file to local path
# Use in analysis framework
GRL_FILE="/path/to/data23_13p6TeV_GRL.xml"
```

**Note:** The agent cannot download GRL files from web URLs. Users must provide local file paths.

### Step 3: Check Luminosity Information

For luminosity calculations in analysis:
- Use iLumiCalc tool for precise luminosity calculations
- Apply GRL XML files (provided by user) to luminosity calculations
- Account for prescales and trigger efficiency

## Best Practices

### Data Quality Flags

**Understanding DQ Flags:**
- **Green flags**: Detector operating normally
- **Yellow flags**: Minor issues, may be usable
- **Red flags**: Serious problems, exclude from analysis
- **Black flags**: Detector off or no data

**Common DQ Tags:**
- `PHYS_StandardGRL_All_Good`: Standard good runs for physics
- `PHYS_StandardGRL_All_Good_25ns`: Good runs with 25ns bunch spacing
- Use specific subsystem flags if your analysis is sensitive to particular detectors

### Run Selection for Analysis

**Typical workflow:**
1. Identify data-taking period (year, period letters)
2. Select appropriate stream (usually `physics_Main`)
3. Obtain and apply standard GRL for physics (user must provide GRL file)
4. Apply additional triggers if needed
5. Calculate luminosity for selected runs

### Trigger Queries

For analyses requiring specific triggers, query via AMI:
```bash
# Query run information including trigger details
ami show run <run_number>
```

## Common Workflows

### Workflow 1: Find Good Runs for Analysis

```bash
# 1. Identify your data period
# Example: 2023 data, periods A-D

# 2. Go to ATLAS Run Query web interface
# https://atlas-runquery.cern.ch/

# 2. User obtains official GRL file from ATLAS Data Preparation
# (agent cannot access Run Query web interface)

# 3. User provides GRL file path to agent

# 4. Use GRL in your analysis framework
# Example in AnalysisTop config:
# GRLDir: GoodRunsLists/
# GRLFile: data23_13p6TeV/20230725/data23_13p6TeV.periodAllYear_DetStatus-v109-pro28-04_MERGED_PHYS_StandardGRL_All_Good.xml
```

### Workflow 2: Check Specific Run Details

```bash
# To investigate a specific run (e.g., run 450123)

# Use AMI to query run information
# Note: User must have already run setupATLAS, lsetup pyami, and voms-proxy-init
ami show run 450123

# Check:
# - Which detectors were active
# - Trigger menu used
# - Luminosity blocks recorded
# - Any known issues or special conditions
```

### Workflow 3: Use GRL for Specific Trigger

```bash
# For analysis requiring specific trigger (e.g., HLT_mu26_ivarmedium)

# 1. User obtains GRL file filtered for specific trigger from ATLAS sources
#    (agent cannot access Run Query web interface to create custom GRLs)

# 2. User provides local GRL file path

# 3. Use custom GRL in analysis
```

## Integration with Other ATLAS Tools

### AMI for Run Queries
- Use AMI command-line tool for run information: `ami show run <run_number>`
- Query dataset metadata (see ami-query skill)
- Cross-reference runs between AMI queries

### GRL Files + Analysis Frameworks

**AnalysisTop:**
```bash
# Use GRL in TopConfig file
GRLDir: GoodRunsLists/
GRLFile: <your_grl>.xml
```

**AthAnalysis:**
```python
# Apply GRL in job options
from GoodRunsLists.GoodRunsListsConf import GoodRunsListSelectorTool
GRLTool = GoodRunsListSelectorTool()
GRLTool.GoodRunsListVec = ['data23_grl.xml']
```

### Run Query + Luminosity Calculation

Use GRLs with iLumiCalc:
```bash
# Calculate luminosity for GRL
iLumiCalc.exe \
  --xml=<your_grl>.xml \
  --begin-run=<first_run> \
  --end-run=<last_run> \
  --trigger=<trigger_name>
```

## Additional Resources

**Note:** The following resources are for human reference only. The agent cannot access these URLs:
- ATLAS Run Query documentation
- Good Runs Lists documentation
- Data Preparation documentation
- Data Quality Flags documentation
- Trigger Information
- Luminosity Public Results

## Common Issues

### GRL File Not Found

If GRL XML files are missing:
- Check if file exists in CVMFS: `/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/GoodRunsLists/`
- Verify the GRL is for correct data-taking period
- User must provide the correct local file path

### Conflicting Run Information

If different tools show conflicting info:
- Official GRLs (user-provided files) are authoritative for data quality
- AMI may have cached/older information
- Consult with user for latest official sources

### Luminosity Calculation Issues

If luminosity numbers don't match:
- Verify correct GRL file is applied
- Check if trigger prescales are accounted for
- Use official iLumiCalc tool for precision
- Compare with user-provided published luminosity results

## Notes

- Data quality flags are updated periodically; users should provide latest GRLs
- Different physics analyses may need different DQ requirements
- Trigger prescales change during data-taking; verify via AMI queries
- Some runs may have partial detector coverage (check via `ami show run`)
- Luminosity uncertainty is typically 1-2% for ATLAS
- Always use official GRLs (user-provided) for publication-quality results
- Agent can only work with command-line tools and local files, not web interfaces
