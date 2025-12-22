---
name: ami-query
description: Query ATLAS Metadata Interface (AMI) for dataset information, production tags, and software configurations on LXPlus
---

# AMI Query Skill

This skill provides guidance for querying the ATLAS Metadata Interface (AMI) to find dataset information, production parameters, and metadata for ATLAS data analysis.

## Description

AMI (ATLAS Metadata Interface) is the central metadata catalog for the ATLAS experiment at CERN. It stores comprehensive metadata for all ATLAS datasets including production status, event numbers, file locations, processing history, and software configurations ("AMI tags"). This skill helps you query AMI effectively from the LXPlus command line.

## When to Use This Skill

Use this skill when you need to:
- Find and list ATLAS datasets by name pattern
- Retrieve detailed metadata about specific datasets
- Look up AMI tags (software configuration bookkeeping)
- Check dataset production status and parameters
- Find file locations and processing history
- Query software container images
- Integrate AMI queries into analysis workflows

## Prerequisites

This skill assumes you are running on the CERN LXPlus cluster with:
- Valid ATLAS computing account
- Access to ATLAS software stack via CVMFS
- Active VOMS proxy for ATLAS
- No additional software installation needed

## How to Use

### Step 1: Setup AMI Environment

Before querying AMI, you need to set up your ATLAS environment and AMI tools:

```bash
# Setup ATLAS environment
setupATLAS

# Initialize VOMS proxy for authentication
voms-proxy-init -voms atlas

# Setup pyAMI command-line tool
lsetup pyami
```

**Note:** The `setupATLAS` and `lsetup` commands are available on LXPlus and other ATLAS computing clusters via CVMFS.

### Step 2: List Datasets by Pattern

To find datasets matching a specific pattern:

```bash
# List all datasets matching a pattern (% is wildcard)
ami list datasets <pattern>

# Examples:
ami list datasets mc20_13TeV.602074%
ami list datasets data23_13p6TeV.%PHYS%
```

**Pattern syntax:**
- `%` acts as a wildcard (matches any characters)
- Use specific run numbers, campaign names, or data types to narrow searches
- ATLAS dataset naming follows: `<datatype>_<energy>.<run_number>.<sample_name>.<format>.<tags>`

### Step 3: Get Detailed Dataset Information

Once you've identified a dataset, retrieve its full metadata:

```bash
# Show detailed information about a specific dataset
ami show dataset info <dataset_name>

# Example:
ami show dataset info mc20_13TeV.602074.PhPy8EG_PDF4LHC21_gg1200NW_ZZ_4lep.deriv.DAOD_PHYS.e8530_s3797_r13167_p6490
```

This returns:
- Dataset name and scope
- Number of events
- File sizes and locations
- Production tags (e, s, r, p tags)
- Parent datasets
- Processing campaign information

### Step 4: Query AMI Tags

AMI tags identify software configurations used in production:

```bash
# Get information about a specific AMI tag
ami show tag info <tag_name>

# Examples:
ami show tag info e8530
ami show tag info r13167
```

Tag types:
- **e-tags**: Event generation parameters
- **s-tags**: Simulation configuration
- **r-tags**: Reconstruction settings
- **p-tags**: Derivation production details

## Best Practices

### Efficient Pattern Matching
- Start with specific patterns to avoid overwhelming results
- Use campaign identifiers (mc20, data23, etc.) to filter by data-taking period
- Combine multiple criteria: `mc20_13TeV.%.<campaign>%.DAOD_PHYS.%`

### Dataset Naming Convention
ATLAS datasets follow this structure:
```
<project>_<energy>.<run_number>.<physics_short>.<format>.<tags>
```

Example breakdown:
- `mc20_13TeV`: MC production campaign, 13 TeV
- `602074`: Dataset ID number
- `PhPy8EG_PDF4LHC21_gg1200NW_ZZ_4lep`: Physics process description
- `deriv.DAOD_PHYS`: Derived format (DAOD_PHYS)
- `e8530_s3797_r13167_p6490`: Production tags

### Common Queries

**Find all DAOD_PHYS datasets for a specific physics sample:**
```bash
ami list datasets mc20_13TeV.%.ttbar%.DAOD_PHYS.%
```

**Find data from 2023 runs:**
```bash
ami list datasets data23_13p6TeV.%
```

**Check derivation production status:**
```bash
ami show dataset info <dataset_name> | grep -i status
```

## Integration with Analysis Workflows

### Using AMI Before Grid Submission

Before submitting jobs to the grid, verify datasets exist and check their properties:

```bash
# 1. Setup environment
setupATLAS
voms-proxy-init -voms atlas
lsetup pyami

# 2. List datasets of interest
ami list datasets mc20_13TeV.602074%

# 3. Get full info on selected dataset
ami show dataset info mc20_13TeV.602074.PhPy8EG_PDF4LHC21_gg1200NW_ZZ_4lep.deriv.DAOD_PHYS.e8530_s3797_r13167_p6490

# 4. Proceed with grid job submission once verified
```

### Scripting AMI Queries

For batch queries, AMI commands can be incorporated into shell scripts:

```bash
#!/bin/bash
setupATLAS
voms-proxy-init -voms atlas -valid 96:00
lsetup pyami

datasets=(
  "mc20_13TeV.602074%"
  "mc20_13TeV.602075%"
)

for pattern in "${datasets[@]}"; do
  echo "Querying: $pattern"
  ami list datasets "$pattern"
done
```

## AMI Web Interface

While this skill focuses on command-line usage, AMI also provides web interfaces:

- **Simple Search**: https://atlas-ami.cern.ch/?subapp=simpleSearch
- **Advanced Search**: https://atlas-ami.cern.ch/?subapp=search
- **AMI Home**: https://atlas-ami.cern.ch/

The web interface is useful for:
- Interactive browsing of datasets
- Multi-criteria searches with filters
- Visualizing dataset relationships

## Additional Resources

- **AMI Documentation**: https://ami.in2p3.fr/docs/pyAMI/
- **AMI Ecosystem Home**: https://ami-ecosystem.in2p3.fr/
- **ATLAS Production Group**: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/AtlasProductionGroup
- **Dataset Naming Conventions**: CERN Document Server (CDS) record 2860920

## Common Issues

### Authentication Errors
If you encounter authentication issues:
```bash
# Renew your VOMS proxy
voms-proxy-destroy
voms-proxy-init -voms atlas -valid 96:00
```

### Command Not Found
If `ami` command is not found:
```bash
# Ensure pyAMI is set up in current shell
lsetup pyami
```

### No Results for Query
- Verify pattern syntax (use `%` for wildcards)
- Check dataset name spelling and campaign identifier
- Try broader patterns initially, then narrow down

## Notes

- AMI queries require valid VOMS proxy (use `voms-proxy-init -voms atlas`)
- Dataset metadata updates periodically; production tags may change
- Use specific p-tags when reproducibility is critical
- Newer p-tags generally supersede older ones (check with derivation team)
- AMI integrates with Rucio (DDM) and PanDA (production system)
