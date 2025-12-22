---
name: atlas-runquery
description: Query ATLAS run information, data quality flags, and work with Good Runs Lists (GRL) for physics analysis on LXPlus
---

# ATLAS Run Query Skill

Query run information, data quality, and work with Good Runs Lists (GRL) for ATLAS physics analysis using command-line tools on LXPlus.

## Prerequisites

**User must complete before starting agent:**
- Run `setupATLAS`, `lsetup pyami` (if using AMI), `voms-proxy-init -voms atlas`
- On LXPlus cluster (verify: `echo $HOSTNAME` should show `lxplus*.cern.ch`)

**Agent limitations:**
- Command-line only (no web access to atlas-runquery.cern.ch or Twiki)
- Cannot download GRL files from URLs - users must provide local file paths
- Uses AMI for run queries, not web interface

## Core Commands

### Query Run Information (AMI)
```bash
# Get run details
ami show run <run_number>

# Example:
ami show run 450123
# Shows: detector status, triggers, luminosity blocks, data streams
```

### Work with GRL Files

**Important:** Agent cannot download GRL files. Users must:
1. Download GRL XML from ATLAS official sources
2. Provide local file path to agent

```bash
# Example GRL usage in analysis
GRL_FILE="/path/to/data23_13p6TeV_GRL.xml"

# GRL files typically in:
/cvmfs/atlas.cern.ch/repo/sw/database/GroupData/GoodRunsLists/
```

## Common Workflows

### Query Specific Run
```bash
# Get run information
ami show run 450123

# Check:
# - Which detectors were active
# - Trigger menu used
# - Luminosity blocks recorded
# - Data quality status
```

### Use GRL in Analysis

**AnalysisTop Example:**
```bash
# In TopConfig file:
# GRLDir: GoodRunsLists/
# GRLFile: data23_13p6TeV/GRL.xml
```

**AthAnalysis Example:**
```python
# In job options:
from GoodRunsLists.GoodRunsListsConf import *
ToolSvc += GoodRunsListSelectorTool(
    "GRLTool",
    GoodRunsListVec = ["/path/to/GRL.xml"]
)
```

### Find Datasets for Good Runs

```bash
# User provides GRL file
# Use file to determine run numbers/periods

# Find corresponding datasets with AMI
ami list datasets data23_13p6TeV.%.DAOD_PHYS.%
```

## Data Quality Flags

**Common DQ Tags:**
- `PHYS_StandardGRL_All_Good`: Standard physics GRL
- `PHYS_StandardGRL_All_Good_25ns`: Good runs with 25ns bunch spacing

**DQ Flag Colors:**
- **Green**: Detector operating normally
- **Yellow**: Minor issues, may be usable
- **Red**: Serious problems, exclude from analysis
- **Black**: Detector off or no data

## Typical Analysis Workflow

1. User obtains official GRL file from ATLAS Data Preparation
2. User provides GRL file path to agent
3. Agent helps configure analysis framework with GRL
4. Use AMI to find datasets matching run periods in GRL
5. Download datasets with Rucio (see rucio-management skill)

## Integration with AMI

```bash
# Query runs for dataset context
ami show run <run_number>

# Find datasets for specific period
ami list datasets data23_13p6TeV.period<X>.%
```

## Luminosity Calculations

For analysis:
- Use iLumiCalc tool with GRL files
- Apply GRL XML to luminosity calculations
- Account for prescales and trigger efficiency

```bash
# Typical usage (exact command varies by release)
iLumiCalc --xml=<GRL.xml> --lumitag=<tag>
```

## Trigger Information

Query trigger details via AMI:
```bash
# Run information includes trigger menu
ami show run <run_number>
```

## Troubleshooting

**GRL File Not Found:**
- Check file path is correct
- Verify file exists in CVMFS if using official GRL
- User must provide valid local path

**Authentication Errors:**
- Check proxy: `voms-proxy-info`
- If expired, user must renew before restarting agent

**AMI Query Fails:**
- Verify `lsetup pyami` was run by user
- Check VOMS proxy is active

## Notes

- Official GRLs are authoritative for data quality
- Different analyses may need different DQ requirements
- Trigger prescales change during data-taking
- Luminosity uncertainty typically 1-2% for ATLAS
- Always use official GRLs for publication results
- Agent works with local GRL files and AMI queries only
