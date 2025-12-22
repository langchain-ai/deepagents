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
- **Active VOMS proxy** (user must run `voms-proxy-init -voms atlas` before launching the agent)
- Access to ATLAS Twiki pages and databases
- No additional software installation needed

**Important**: The user must set up the ATLAS environment and VOMS proxy **before** starting the agent session. If these are not set up, AMI queries will fail with authentication errors.

## Available Tools for Run Queries

ATLAS provides several tools for querying run information:

### 1. ATLAS Run Query Web Interface

**Primary Tool**: ATLAS Run Query Page
- **URL**: https://atlas-runquery.cern.ch/
- **Use for**: Interactive browsing and filtering of runs
- **Features**: 
  - Multi-criteria search (run number, period, stream, trigger)
  - Data quality flag filtering
  - Luminosity calculations
  - Export results as XML, JSON, or text

### 2. AMI (ATLAS Metadata Interface)

AMI also contains run-level information:
```bash
# Setup pyAMI if not already done
lsetup pyami

# Query runs in AMI
ami show run <run_number>
```

**Note**: If you see authentication errors, the user's VOMS proxy may have expired. The user must run `voms-proxy-init -voms atlas` in their shell before restarting the agent.

### 3. Good Runs Lists (GRL)

GRLs are pre-computed lists of runs passing data quality requirements:
- Published by ATLAS Data Preparation group
- Available in XML format
- Used in analysis frameworks (AnalysisTop, AthAnalysis)

## How to Use

### Step 1: Access Run Query Web Interface

The primary method for run queries is the web interface:

1. Navigate to: https://atlas-runquery.cern.ch/
2. Authenticate with your CERN account
3. Use the search interface to filter runs

### Step 2: Filter Runs by Criteria

**Common search criteria:**

**By Run Number or Range:**
```
Run number: 450000-460000
```

**By Data Period:**
```
Period: period A, period B
Year: 2023, 2024
```

**By Stream:**
```
Stream: physics_Main, physics_ZeroBias, express_express
```

**By Data Quality:**
```
Data quality: PHYS_StandardGRL_All_Good
```

### Step 3: Query Specific Run Information

For detailed information about a specific run:

**Via Web Interface:**
1. Go to https://atlas-runquery.cern.ch/
2. Enter run number in search box
3. View detailed run information:
   - Detector status (subsystems on/off)
   - Trigger configuration
   - Luminosity delivered/recorded
   - Data streams available
   - Data quality assessments

**Via AMI:**
```bash
setupATLAS
lsetup pyami
voms-proxy-init -voms atlas

# Get run information
ami show run <run_number>
```

### Step 4: Generate Good Runs Lists

For physics analysis, you need runs with good data quality:

**Using Official GRLs:**

Official GRLs are published at:
- **Twiki**: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/GoodRunLists

Download appropriate GRL XML file for your analysis:
```bash
# Example for Run 3 data
wget https://atlas-runquery.cern.ch/GRL/.../<GRL_file>.xml
```

**Creating Custom GRLs:**

Use the Run Query interface to create custom GRLs:
1. Apply desired filters (period, triggers, DQ flags)
2. Select matching runs
3. Export as XML format
4. Use in your analysis framework

### Step 5: Check Luminosity Information

Query integrated luminosity for run selections:

**Via Run Query Interface:**
1. Select runs using filters
2. View "Luminosity" column in results
3. Sum luminosity for selected runs

**For Analysis:**
- Use iLumiCalc tool for precise luminosity calculations
- Apply GRL XML files to luminosity calculations
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
3. Apply standard GRL for physics
4. Apply additional triggers if needed
5. Calculate luminosity for selected runs

### Trigger Queries

For analyses requiring specific triggers:
1. Search by trigger name in Run Query
2. Verify trigger was active and unprescaled
3. Check trigger rates and efficiency
4. Account for prescales in luminosity calculation

## Common Workflows

### Workflow 1: Find Good Runs for Analysis

```bash
# 1. Identify your data period
# Example: 2023 data, periods A-D

# 2. Go to ATLAS Run Query web interface
# https://atlas-runquery.cern.ch/

# 3. Apply filters:
#    - Year: 2023
#    - Period: A, B, C, D
#    - Stream: physics_Main
#    - DQ: PHYS_StandardGRL_All_Good

# 4. Download official GRL or export custom list

# 5. Use GRL in your analysis framework
# Example in AnalysisTop config:
# GRLDir: GoodRunsLists/
# GRLFile: data23_13p6TeV/20230725/data23_13p6TeV.periodAllYear_DetStatus-v109-pro28-04_MERGED_PHYS_StandardGRL_All_Good.xml
```

### Workflow 2: Check Specific Run Details

```bash
# To investigate a specific run (e.g., run 450123)

# Option 1: Web interface
# https://atlas-runquery.cern.ch/
# Search: run 450123
# Review: detector status, triggers, luminosity

# Option 2: AMI
# Note: User must have already initialized ATLAS environment and VOMS proxy
lsetup pyami
ami show run 450123

# Check:
# - Which detectors were active
# - Trigger menu used
# - Luminosity blocks recorded
# - Any known issues or special conditions
```

### Workflow 3: Create GRL for Specific Trigger

```bash
# For analysis requiring specific trigger (e.g., HLT_mu26_ivarmedium)

# 1. Go to ATLAS Run Query
# 2. Search criteria:
#    - Year: 2023
#    - Trigger: HLT_mu26_ivarmedium
#    - DQ: PHYS_StandardGRL_All_Good
# 3. Verify trigger was unprescaled (prescale = 1)
# 4. Export matching runs as XML
# 5. Use custom GRL in analysis
```

## Integration with Other ATLAS Tools

### Run Query + AMI
- Use Run Query for data quality and run selection
- Use AMI for detailed dataset metadata (see ami-query skill)
- Cross-reference runs between tools

### Run Query + Analysis Frameworks

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

- **ATLAS Run Query**: https://atlas-runquery.cern.ch/
- **Good Runs Lists**: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/GoodRunLists
- **Data Preparation Twiki**: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/DataPreparation
- **Data Quality Flags**: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/DataQuality
- **Trigger Information**: https://twiki.cern.ch/twiki/bin/view/Atlas/TriggerMenu
- **Luminosity Public Results**: https://twiki.cern.ch/twiki/bin/view/AtlasPublic/LuminosityPublicResults

## Common Issues

### Cannot Access Run Query Web Interface

If you cannot access the run query page:
- Verify you're logged in with CERN credentials
- Check you have ATLAS computing account
- Ensure you're on CERN network or using VPN
- Try alternative: use AMI for run queries

### GRL File Not Found

If GRL XML files are missing:
- Check official GRL repository: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/GoodRunLists
- Verify the GRL is for correct data-taking period
- Ensure you have correct CVMFS access: `/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/GoodRunsLists/`

### Conflicting Run Information

If different tools show conflicting info:
- Official GRLs are authoritative for data quality
- Run Query web interface is most up-to-date
- AMI may have cached/older information
- Consult Data Preparation group for clarifications

### Luminosity Calculation Issues

If luminosity numbers don't match:
- Verify correct GRL is applied
- Check if trigger prescales are accounted for
- Use official iLumiCalc tool for precision
- Compare with published luminosity results

## Notes

- Data quality flags are updated periodically; use latest GRLs
- Different physics analyses may need different DQ requirements
- Trigger prescales change during data-taking; verify for your runs
- Some runs may have partial detector coverage (check detector status)
- Luminosity uncertainty is typically 1-2% for ATLAS
- Always use official GRLs for publication-quality results
- Check ATLAS Physics Plenary for latest GRL recommendations
