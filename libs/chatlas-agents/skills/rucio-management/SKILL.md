---
name: rucio-management
description: Download and manage ATLAS grid data using Rucio distributed data management (DDM) system on LXPlus
---

# Rucio Management Skill

Manage ATLAS grid data using Rucio: download datasets, create replication rules, and manage grid job outputs from LXPlus command line.

## Prerequisites

**User must complete before starting agent:**
- Run `setupATLAS`, `localSetupRucioClients`, `voms-proxy-init -voms atlas`
- On LXPlus cluster (verify: `echo $HOSTNAME` should show `lxplus*.cern.ch`)
- Storage quota on appropriate disk areas

**Common errors if not set up:**
- `bash: rucio: command not found` → User must run `localSetupRucioClients`
- `ERROR: No valid proxy found` → User must run `voms-proxy-init -voms atlas`

**Agent limitations:**
- Command-line only (no web access to rucio-ui.cern.ch or bigpanda.cern.ch)
- Cannot access BigPanDA - users must provide container names from their grid jobs
- Cannot download files from web URLs

## Core Commands

### List Dataset Contents
```bash
# List files in dataset
rucio list-files <dataset_name>

# With details (size, checksum, location)
rucio list-files --csv <dataset_name>
```

### Download Datasets
```bash
# Download with verbose output
rucio -v download <dataset_name>

# Download to specific directory
rucio download --dir /path/to/output <dataset_name>

# Example:
rucio -v download mc20_13TeV:mc20_13TeV.602074.DAOD_PHYS.p6490
```

### Check Data Locations
```bash
# List dataset replicas
rucio list-dataset-replicas <dataset_name>

# File-level replica locations
rucio list-file-replicas <dataset_name>
```

### Create Replication Rules
```bash
# Preserve data (prevent deletion after ~2 weeks)
rucio add-rule <dataset_name> 1 <RSE>

# Examples:
rucio add-rule user.jsmith:output.root 1 CERN-PROD_USERDISK
rucio add-rule mc20_13TeV:dataset 1 CERN-PROD_DATADISK

# Verify rule
rucio list-rules <dataset_name>
```

### Check Quotas
```bash
# Check account usage
rucio list-account-usage <username> <RSE>

# Example:
rucio list-account-usage jsmith CERN-PROD_USERDISK
```

## Common Workflows

### Download Grid Job Output
```bash
# User provides container name from their PanDA job
# (agent cannot access BigPanDA web interface)

# 1. List files in container
rucio list-files user.jsmith:user.jsmith.12345678._000001.output_h5

# 2. Download
rucio -v download user.jsmith:user.jsmith.12345678._000001.output_h5

# 3. Create preservation rule (optional)
rucio add-rule user.jsmith:user.jsmith.12345678._000001.output_h5 1 CERN-PROD_USERDISK
```

### Download MC Dataset
```bash
# 1. Find dataset with AMI (see ami-query skill)
ami list datasets mc20_13TeV.602074.%.DAOD_PHYS.%

# 2. Check replica locations
rucio list-dataset-replicas mc20_13TeV:mc20_13TeV.602074.DAOD_PHYS.p6490

# 3. Download to analysis area
rucio download --dir $HOME/analysis/data mc20_13TeV:mc20_13TeV.602074.DAOD_PHYS.p6490
```

### Preserve Analysis Output
```bash
# User provides container name from their analysis

# Create replication rule
rucio add-rule user.jsmith:user.jsmith.results_root 1 CERN-PROD_USERDISK

# Verify rule creation
rucio list-rules user.jsmith:user.jsmith.results_root

# Download when ready
rucio -v download user.jsmith:user.jsmith.results_root
```

## Dataset Naming

Format: `scope:dataset_name`
- **scope**: Username (user.jsmith), project (mc20_13TeV, data23_13p6TeV)
- **dataset_name**: Full dataset identifier

Examples:
- `mc20_13TeV:mc20_13TeV.602074.PhPy8EG.DAOD_PHYS.p6490`
- `user.jsmith:user.jsmith.12345678._000001.output_h5`
- `data23_13p6TeV:data23_13p6TeV.00450123.physics_Main.DAOD_PHYS.p6490`

## Integration with Other Tools

**AMI + Rucio Pipeline:**
1. Use AMI to find datasets
2. Use Rucio to download

**Analysis Frameworks:**
```bash
# Download for AnalysisTop, AthAnalysis, etc.
rucio download --dir $TestArea/data <dataset>
```

## Troubleshooting

**Authentication Errors:**
- Check proxy: `voms-proxy-info`
- If expired, user must renew before restarting agent

**Download Stalls:**
```bash
# Try with --allow-tape
rucio download --allow-tape <dataset_name>

# Or specify RSE
rucio download --rse CERN-PROD_DATADISK <dataset_name>
```

**Quota Exceeded:**
```bash
# Check usage
rucio list-account-usage <username> CERN-PROD_USERDISK
# Clean up old data or request quota increase
```

## Notes

- Grid scratch files deleted after ~2 weeks - use replication rules to preserve
- Use `--allow-tape` if downloads fail (may be slower)
- Check quota before large downloads
- CERN-PROD_USERDISK for user data, CERN-PROD_DATADISK for official datasets
