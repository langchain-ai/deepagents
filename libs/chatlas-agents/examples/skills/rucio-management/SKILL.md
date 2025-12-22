---
name: rucio-management
description: Manage ATLAS grid data and samples with Rucio for dataset download, replication, and distributed data management on LXPlus
---

# Rucio Management Skill

This skill provides guidance for using Rucio, the ATLAS Distributed Data Management (DDM) system, to download datasets, manage data replication rules, and interact with grid storage from LXPlus.

## Description

Rucio is the distributed data management system used by ATLAS for managing petabytes of data across the Worldwide LHC Computing Grid (WLCG). It handles dataset replication, file catalog services, and data transfers. This skill helps you use Rucio effectively for accessing and managing ATLAS data.

## When to Use This Skill

Use this skill when you need to:
- Download datasets from the grid to local storage
- List files in a dataset or container
- Check data locations and replica availability
- Create replication rules to preserve datasets
- Manage dataset subscriptions and quotas
- Query file metadata and checksums
- Download output files from grid jobs
- Transfer data between storage elements

## Prerequisites

This skill assumes you are running on the CERN LXPlus cluster with:
- Valid ATLAS computing account
- Access to ATLAS software stack via CVMFS
- **ATLAS environment already initialized** (user must run `setupATLAS` before launching the agent)
- **Rucio clients already set up** (user must run `localSetupRucioClients` before launching the agent)
- **Active VOMS proxy** (user must run `voms-proxy-init -voms atlas` before launching the agent)
- Storage quota on appropriate disk areas
- No additional software installation needed

**Important**: The user must set up the ATLAS environment, Rucio clients, and VOMS proxy **before** starting the agent session. If these are not set up, Rucio commands will fail with authentication or command-not-found errors.

**Note:** On some non-LXPlus clusters, you may also need to run `lsetup emi` before starting the agent.

## How to Use

### Step 1: Find Dataset Names

**Common Errors if Prerequisites Not Met:**

If you see errors like:
- `bash: rucio: command not found` → Rucio clients not set up. User must run `localSetupRucioClients` in their shell before starting the agent.
- `ERROR: No valid proxy found` or `Authentication failed` → VOMS proxy expired or not initialized. User must run `voms-proxy-init -voms atlas` in their shell before starting the agent.
- `setupATLAS: command not found` → Not on LXPlus or ATLAS environment not sourced. User must run `setupATLAS` in their shell before starting the agent.

### Finding Dataset Names

Before downloading, you need the full dataset/container name. This typically comes from:
- AMI queries (use the ami-query skill)
- Grid job output (check PanDA BigPanDA webpage)
- Analysis documentation or colleagues

Dataset names follow the pattern:
```
scope:dataset_name
```

Example:
```
user.jsmith:user.jsmith.myanalysis.output.h5
mc20_13TeV:mc20_13TeV.602074.PhPy8EG.DAOD_PHYS.e8530_s3797_r13167_p6490
```

### Step 2: List Dataset Contents

Check what files are in a dataset or container:

```bash
# List files in a dataset
rucio list-files <dataset_name>

# List with more details (size, checksum, storage element)
rucio list-files --csv <dataset_name>

# Example:
rucio list-files user.jsmith:user.jsmith.myanalysis.output_h5
```

### Step 3: Download Datasets

Download dataset files to your local directory:

```bash
# Download with verbose output
rucio -v download <dataset_name>

# Download to specific directory
rucio download --dir /path/to/output <dataset_name>

# Examples:
rucio -v download user.jsmith:user.jsmith.myanalysis.output_h5
rucio download --dir $HOME/data user.jsmith:user.jsmith.myanalysis.output_h5
```

**Download behavior:**
- Creates subdirectory with scope and dataset name
- Verifies checksums automatically
- Skips already-downloaded files
- Shows progress and transfer rates

### Step 4: Check Data Locations

Find where dataset replicas are stored on the grid:

```bash
# List all replicas of a dataset
rucio list-dataset-replicas <dataset_name>

# Show file-level replica locations
rucio list-file-replicas <dataset_name>

# Example:
rucio list-dataset-replicas mc20_13TeV:mc20_13TeV.602074.DAOD_PHYS.p6490
```

### Step 5: Create Replication Rules (Preserve Data)

Grid files on scratch disks are typically deleted after ~2 weeks. To preserve important data, create a replication rule:

```bash
# Create rule to copy dataset to a storage element where you have quota
rucio add-rule <dataset_name> 1 <RSE>

# Examples:
rucio add-rule user.jsmith:user.jsmith.myanalysis.output_h5 1 CERN-PROD_LOCALGROUPDISK
rucio add-rule user.jsmith:user.jsmith.myanalysis.output_h5 1 CERN-PROD_USERDISK
```

**Common RSEs (Rucio Storage Elements):**
- `CERN-PROD_USERDISK`: User storage at CERN
- `CERN-PROD_LOCALGROUPDISK`: Local group disk at CERN
- `CERN-PROD_SCRATCHDISK`: Temporary scratch storage (auto-deleted)

**Note:** Check your quota and available space before creating rules.

## Best Practices

### Data Lifecycle Management

**Scratch vs Permanent Storage:**
- **Scratch disks**: Temporary storage, files auto-deleted after ~2 weeks
- **Permanent disks**: Long-term storage (USERDISK, LOCALGROUPDISK)
- Always create replication rules for important outputs

### Efficient Downloads

**Download only what you need:**
```bash
# Download specific files from a container
rucio list-files user.jsmith:mycontainer | grep "pattern" | awk '{print $2}' | xargs -I {} rucio download {}
```

**Check size before downloading:**
```bash
rucio list-dataset-replicas <dataset_name> | grep "TOTAL"
```

### Quota Management

Check your quota usage:
```bash
# List your account quotas
rucio list-account-limits <your_username>

# Check usage at specific RSE
rucio list-account-usage <your_username> <RSE>
```

### Working with Grid Job Output

After submitting jobs via PanDA, retrieve outputs using Rucio:

**Step-by-step:**
1. Go to BigPanDA: https://bigpanda.cern.ch/
2. Navigate to "My BigPanDA" → find your task
3. Click task name → scroll to "Containers" section
4. Copy the output container name (ending in `_output.h5` or similar)
5. Use Rucio to download:

```bash
# Note: User must have already run setupATLAS, localSetupRucioClients, and voms-proxy-init
rucio -v download <container_name_from_panda>
```

## Common Workflows

### Workflow 1: Download Grid Job Output

```bash
# Note: User must have already run setupATLAS, localSetupRucioClients, and voms-proxy-init

# 1. Get container name from PanDA webpage
# Example: user.jsmith:user.jsmith.12345678._000001.output_h5

# 2. Check dataset info
rucio list-files user.jsmith:user.jsmith.12345678._000001.output_h5

# 3. Download dataset
rucio -v download user.jsmith:user.jsmith.12345678._000001.output_h5

# 4. Create preservation rule (if needed)
rucio add-rule user.jsmith:user.jsmith.12345678._000001.output_h5 1 CERN-PROD_USERDISK
```

### Workflow 2: Download MC Datasets for Analysis

```bash
# 1. Find dataset using AMI (see ami-query skill)
# Note: User must have already run setupATLAS, lsetup pyami, localSetupRucioClients, and voms-proxy-init
ami list datasets mc20_13TeV.602074.%.DAOD_PHYS.%

# 2. Check dataset size and location
rucio list-dataset-replicas mc20_13TeV:mc20_13TeV.602074.DAOD_PHYS.p6490

# 3. Download to analysis area
rucio download --dir $HOME/analysis/data mc20_13TeV:mc20_13TeV.602074.DAOD_PHYS.p6490
```

### Workflow 3: Preserve Analysis Output

```bash
# After generating important results on grid:
# Note: User must have already run setupATLAS, localSetupRucioClients, and voms-proxy-init

# 1. Identify container from PanDA
# Container: user.jsmith:user.jsmith.myanalysis.results_root

# 2. Create replication rule to permanent storage
rucio add-rule user.jsmith:user.jsmith.myanalysis.results_root 1 CERN-PROD_USERDISK

# 3. Verify rule creation
rucio list-rules user.jsmith:user.jsmith.myanalysis.results_root

# 4. Download when ready
rucio -v download user.jsmith:user.jsmith.myanalysis.results_root
```

## Rucio Web Interface

For interactive data management, use the Rucio web UI:

**Rucio WebUI**: https://rucio-ui.cern.ch/

Features:
- Browse datasets and containers
- Create replication rules with GUI
- Monitor data transfers
- Check quota usage
- View rule status and history

## Integration with Other ATLAS Tools

### Rucio + AMI
1. Use AMI to find datasets (ami-query skill)
2. Use Rucio to download found datasets

### Rucio + PanDA
1. Submit jobs via PanDA
2. Monitor on BigPanDA: https://bigpanda.cern.ch/
3. Get output container names from PanDA
4. Download outputs with Rucio

### Rucio + Analysis Frameworks
```bash
# Download datasets for use in AnalysisTop, AthAnalysis, etc.
rucio download --dir $TestArea/data mc20_13TeV:<dataset>
```

## Additional Resources

- **Rucio Documentation**: https://rucio.cern.ch/documentation/
- **ATLAS DDM Twiki**: https://twiki.cern.ch/twiki/bin/view/AtlasComputing/DataManagement
- **Rucio Commands Reference**: https://rucio.cern.ch/documentation/cli_examples
- **BigPanDA**: https://bigpanda.cern.ch/

## Common Issues

### Authentication Errors

If you get authentication errors:
```bash
# Check VOMS proxy status
voms-proxy-info

# If expired, user must renew proxy in their shell (before restarting agent):
# voms-proxy-init -voms atlas -valid 96:00
```

**Note**: The agent cannot run `voms-proxy-init` - this must be done by the user in their shell before starting the agent session.

### Download Stalls or Fails

If downloads fail or stall:
```bash
# Try with --allow-tape (may be slower)
rucio download --allow-tape <dataset_name>

# Or specify a specific RSE
rucio download --rse CERN-PROD_DATADISK <dataset_name>
```

### Quota Exceeded

If you exceed quota:
```bash
# Check current usage
rucio list-account-usage <your_username> CERN-PROD_USERDISK

# Clean up old data or request quota increase from your group
```

### Dataset Not Found

If Rucio can't find a dataset:
- Verify the full dataset name (scope:name)
- Check if it's a container vs dataset
- Use AMI to verify dataset exists
- Check if you have permissions (some datasets are restricted)

## Notes

- All Rucio commands require active VOMS proxy
- Dataset names are case-sensitive
- Scratch disk files are deleted after ~2 weeks without a preservation rule
- Download speeds depend on network and storage element load
- Use `rucio --help` for full command reference
- Check storage quotas before creating large replication rules
- Rucio integrates with AMI for metadata and PanDA for job outputs
