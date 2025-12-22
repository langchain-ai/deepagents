# ATLAS Software Tools Skills

This directory contains specialized skills for working with ATLAS experiment software tools on the CERN LXPlus cluster. These skills provide guidance for common ATLAS data management and analysis tasks.

## Available ATLAS Skills

### 1. AMI Query (`ami-query`)

**Purpose:** Query the ATLAS Metadata Interface (AMI) for dataset information and metadata.

**Use when you need to:**
- Find and list ATLAS datasets by name pattern
- Retrieve detailed metadata about datasets
- Look up AMI tags (software configuration bookkeeping)
- Check dataset production status and parameters
- Find file locations and processing history

**Key Commands:**
```bash
setupATLAS
voms-proxy-init -voms atlas
lsetup pyami
ami list datasets <pattern>
ami show dataset info <dataset_name>
```

**Learn more:** [ami-query/SKILL.md](./ami-query/SKILL.md)

---

### 2. Rucio Management (`rucio-management`)

**Purpose:** Manage ATLAS grid data using Rucio distributed data management system.

**Use when you need to:**
- Download datasets from the grid to local storage
- List files in a dataset or container
- Check data locations and replica availability
- Create replication rules to preserve datasets
- Download output files from grid jobs

**Key Commands:**
```bash
setupATLAS
localSetupRucioClients
voms-proxy-init -voms atlas
rucio -v download <dataset_name>
rucio add-rule <dataset_name> 1 <RSE>
```

**Learn more:** [rucio-management/SKILL.md](./rucio-management/SKILL.md)

---

### 3. ATLAS Run Query (`atlas-runquery`)

**Purpose:** Query ATLAS run information, data quality, and luminosity records.

**Use when you need to:**
- Find runs from specific data-taking periods
- Check data quality flags for analysis
- Query luminosity information
- Identify runs with specific trigger configurations
- Create Good Runs Lists (GRLs) for analysis

**Key Resources:**
- Web Interface: https://atlas-runquery.cern.ch/
- Good Runs Lists: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/GoodRunLists

**Learn more:** [atlas-runquery/SKILL.md](./atlas-runquery/SKILL.md)

---

## Using These Skills with DeepAgents

### Installation

These skills are available in the `examples/skills` directory of the deepagents-cli package. You can copy them to your user skills directory or use them from the project directory.

### Option 1: Copy to User Skills Directory

```bash
# Create user skills directory if it doesn't exist
mkdir -p ~/.deepagents/agent/skills

# Copy ATLAS skills
cp -r libs/deepagents-cli/examples/skills/ami-query ~/.deepagents/agent/skills/
cp -r libs/deepagents-cli/examples/skills/rucio-management ~/.deepagents/agent/skills/
cp -r libs/deepagents-cli/examples/skills/atlas-runquery ~/.deepagents/agent/skills/

# Verify skills are available
deepagents skills list
```

### Option 2: Use Project-Level Skills

```bash
# Create project skills directory
mkdir -p .deepagents/skills

# Copy ATLAS skills to project
cp -r libs/deepagents-cli/examples/skills/ami-query .deepagents/skills/
cp -r libs/deepagents-cli/examples/skills/rucio-management .deepagents/skills/
cp -r libs/deepagents-cli/examples/skills/atlas-runquery .deepagents/skills/

# Skills will be automatically available when running deepagents in this project
```

### Using Skills in DeepAgents

Once installed, the skills are automatically available to your DeepAgents. The agent will:

1. **Discover skills** at startup (names and descriptions appear in system prompt)
2. **Load skills on-demand** when relevant to a task (reads full SKILL.md)
3. **Follow skill instructions** to complete tasks using ATLAS tools

**Example interaction:**
```
User: "Find all DAOD_PHYS datasets from mc20 campaign for the Higgs to ZZ sample"

Agent: [Recognizes ami-query skill applies]
       [Reads ami-query/SKILL.md for instructions]
       [Executes AMI commands as described in skill]
       [Returns dataset list]
```

## Common Workflows

### Workflow 1: Find and Download Dataset

1. **Find dataset** using AMI:
   ```bash
   ami list datasets mc20_13TeV.%.ZZ%.DAOD_PHYS.%
   ```

2. **Download dataset** using Rucio:
   ```bash
   rucio -v download mc20_13TeV:<selected_dataset>
   ```

### Workflow 2: Download Grid Job Output

1. **Get output container** from PanDA (https://bigpanda.cern.ch/)

2. **Download** using Rucio:
   ```bash
   rucio -v download user.username:<container_name>
   ```

3. **Preserve** important outputs:
   ```bash
   rucio add-rule user.username:<container_name> 1 CERN-PROD_USERDISK
   ```

### Workflow 3: Create Analysis Dataset Selection

1. **Query runs** using Run Query:
   - Web: https://atlas-runquery.cern.ch/
   - Apply data quality filters
   - Export GRL XML

2. **Find datasets** for selected runs:
   ```bash
   ami list datasets data23_13p6TeV.%.DAOD_PHYS.%
   ```

3. **Download** for analysis:
   ```bash
   rucio download --dir $HOME/analysis/data data23_13p6TeV:<dataset>
   ```

## Prerequisites

All ATLAS skills assume:
- **Environment:** CERN LXPlus cluster or ATLAS computing node
- **Account:** Valid ATLAS computing account
- **Authentication:** Active VOMS proxy (`voms-proxy-init -voms atlas`)
- **Software:** ATLAS software stack via CVMFS (no installation needed)

## Skill Development

### Anatomy of a Skill

Each skill consists of:
1. **SKILL.md** - Main documentation with YAML frontmatter
2. **Supporting files** (optional) - Scripts, configs, reference docs

### YAML Frontmatter

Required fields:
```yaml
---
name: skill-name
description: Brief description of what the skill does
---
```

### Skill Content Structure

Recommended sections:
- **Description** - What the skill does
- **When to Use** - Scenarios where skill applies
- **Prerequisites** - Requirements and setup
- **How to Use** - Step-by-step instructions
- **Best Practices** - Tips and recommendations
- **Common Workflows** - End-to-end examples
- **Resources** - Links to documentation
- **Common Issues** - Troubleshooting

## Contributing

To add new ATLAS skills:

1. Create skill directory: `mkdir -p examples/skills/my-atlas-skill`
2. Create SKILL.md with frontmatter and instructions
3. Test with: `python3 -m deepagents_cli.skills.load`
4. Submit PR with new skill

## Additional Resources

### ATLAS Documentation
- **Computing Documentation:** https://twiki.cern.ch/twiki/bin/view/AtlasComputing/
- **Data Management:** https://twiki.cern.ch/twiki/bin/view/AtlasComputing/DataManagement
- **Production System:** https://twiki.cern.ch/twiki/bin/view/AtlasProtected/AtlasProductionGroup

### Training Materials
- **ATLAS Software Tutorial:** https://atlassoftwaredocs.web.cern.ch/
- **Grid Training:** https://twiki.cern.ch/twiki/bin/view/AtlasComputing/GridTraining

### Support
- **Computing Help:** atlas-adc-computing-help@cern.ch
- **Data Management:** atlas-adc-ddm-support@cern.ch
- **Mattermost:** ATLAS Computing Support channels

## License

These skills are provided as examples for the ATLAS computing community. They follow the same license as the deepagents-cli package.

## Version History

- **v1.0** (2024-12) - Initial release with ami-query, rucio-management, atlas-runquery skills
