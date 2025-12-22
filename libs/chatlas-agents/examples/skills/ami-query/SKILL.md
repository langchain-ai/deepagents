---
name: ami-query
description: Query ATLAS Metadata Interface (AMI) for dataset information, production tags, and software configurations on LXPlus
---

# AMI Query Skill

Query the ATLAS Metadata Interface (AMI) for dataset metadata, production parameters, and AMI tags from LXPlus command line.

## Prerequisites

**User must complete before starting agent:**
- Run `setupATLAS`, `lsetup pyami`, `voms-proxy-init -voms atlas`
- On LXPlus cluster (verify: `echo $HOSTNAME` should show `lxplus*.cern.ch`)

**Common errors if not set up:**
- `bash: ami: command not found` → User must run `lsetup pyami`
- `ERROR: No credentials found` → User must run `voms-proxy-init -voms atlas`

**Agent limitations:**
- Command-line only (no web access to atlas-ami.cern.ch or Twiki)
- Cannot access ATLAS internal websites

## Core Commands

### List Datasets
```bash
# Find datasets matching pattern (% is wildcard)
ami list datasets <pattern>

# Examples:
ami list datasets mc20_13TeV.602074%
ami list datasets data23_13p6TeV.%PHYS%
```

### Dataset Details
```bash
# Get comprehensive dataset information
ami show dataset info <dataset_name>

# Example:
ami show dataset info mc20_13TeV.602074.DAOD_PHYS.p6490
```

### Query by AMI Tag
```bash
# List datasets with specific p-tag
ami list datasets %p6490

# Show tag details
ami show tag info p6490
```

### File Locations
```bash
# List files in dataset
ami list files <dataset_name>
```

## Common Workflows

### Find Monte Carlo Dataset
```bash
# 1. Search by physics process
ami list datasets mc20_13TeV.%.Higgs*ZZ%.DAOD_PHYS.%

# 2. Get details of selected dataset
ami show dataset info <selected_dataset>

# 3. Check file count and size
ami list files <selected_dataset>
```

### Find Data Derivation
```bash
# List derivations for specific p-tag
ami list datasets data23_13p6TeV.%.DAOD_PHYS.p6490

# Get provenance info
ami show dataset prov <dataset_name>
```

### Check Production Status
```bash
# Show dataset details including status
ami show dataset info <dataset_name>
# Look for: nEvents, totalEvents, status fields
```

## Integration with Rucio

After finding datasets with AMI, download with Rucio:
```bash
# 1. Find dataset
ami list datasets mc20_13TeV.%.signal%.DAOD_PHYS.%

# 2. Download with Rucio (see rucio-management skill)
rucio -v download mc20_13TeV:<dataset_name>
```

## Troubleshooting

**Authentication Errors:**
- Check VOMS proxy: `voms-proxy-info`
- If expired, user must renew before restarting agent

**No Results:**
- Verify pattern syntax (use `%` for wildcards)
- Try broader patterns, then narrow down
- Check dataset name spelling

**Command Not Found:**
- Agent cannot run `lsetup pyami` - user must do this before starting

## Notes

- Dataset metadata updates periodically
- Use specific p-tags for reproducibility
- Newer p-tags generally supersede older ones
- AMI integrates with Rucio for data downloads
