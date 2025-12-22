# ATLAS Software Tools as DeepAgents Skills - Implementation Summary

## Overview

This document summarizes the implementation of ATLAS software tools as DeepAgents skills, providing agents with structured guidance for working with ATLAS data management and analysis tools on the CERN LXPlus cluster.

## Research Phase

### ChATLAS MCP Search Results

Used the `search_chatlas` MCP tool to gather information about ATLAS software tools:

#### AMI (ATLAS Metadata Interface)
- **Purpose**: Central metadata catalog for ATLAS datasets
- **Command-line tool**: `pyami` (accessed via `lsetup pyami`)
- **Key features**:
  - Dataset browsing and search
  - Production tags (AMI tags) bookkeeping
  - Software container image storage
  - Integration with PanDA and DDM
- **Usage**: `ami list datasets <pattern>`, `ami show dataset info <dataset>`
- **Documentation**: https://ami.in2p3.fr/docs/pyAMI/

#### Rucio (Grid Data Management)
- **Purpose**: Distributed data management system for ATLAS
- **Command-line tool**: Rucio clients (accessed via `localSetupRucioClients`)
- **Key features**:
  - Dataset download from grid storage
  - Replica management
  - Data replication rules (preservation)
  - File catalog services
- **Usage**: `rucio download <dataset>`, `rucio add-rule <dataset> 1 <RSE>`
- **Common workflow**: Download grid job outputs, preserve data on permanent storage
- **Documentation**: https://rucio.cern.ch/documentation/

#### Run Query
- **Purpose**: Query run information, data quality, and luminosity
- **Tools**:
  - Web interface: https://atlas-runquery.cern.ch/
  - AMI integration for run metadata
  - Good Runs Lists (GRLs) for analysis
- **Key features**:
  - Data quality flag filtering
  - Trigger configuration queries
  - Luminosity calculations
  - Run selection for physics analysis
- **Documentation**: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/GoodRunLists

### LangChain Docs on Skills Feature

Researched the skills feature in deepagents using `langchain_docs-SearchDocsByLangChain`:

#### Skills Architecture
- **Pattern**: Progressive disclosure of specialized capabilities
- **Structure**: YAML frontmatter + markdown instructions
- **Discovery**: Skills injected into system prompt (name + description)
- **Loading**: Full SKILL.md read on-demand when relevant to task
- **Implementation**: SkillsMiddleware in deepagents-cli

#### Skill Components
1. **SKILL.md** - Required file with:
   - YAML frontmatter (name, description)
   - Detailed markdown instructions
   - Examples and best practices

2. **Supporting files** (optional):
   - Python scripts
   - Configuration files
   - Reference documentation

#### Progressive Disclosure Pattern
1. Agent sees skill names and descriptions in system prompt
2. Agent recognizes when skill applies to user's task
3. Agent reads full SKILL.md for detailed instructions
4. Agent follows skill's step-by-step workflow

## Implementation

### Created ATLAS Skills

Implemented three comprehensive skills in `libs/chatlas-agents/skills/`:

#### 1. AMI Query (`ami-query/`)
**File**: `ami-query/SKILL.md`

**Content structure**:
- **YAML frontmatter**: Name, description for discovery
- **Description**: AMI overview and capabilities
- **When to Use**: Scenarios (find datasets, check metadata, query tags)
- **Prerequisites**: LXPlus, VOMS proxy, CVMFS access
- **How to Use**: Step-by-step instructions
  - Setup AMI environment (`setupATLAS`, `lsetup pyami`)
  - List datasets by pattern
  - Get detailed dataset information
  - Query AMI tags
- **Best Practices**: Pattern matching, naming conventions
- **Integration**: Workflows for grid submission, scripting
- **Resources**: Links to AMI docs, ATLAS twikis
- **Common Issues**: Troubleshooting section

**Key commands covered**:
```bash
setupATLAS
voms-proxy-init -voms atlas
lsetup pyami
ami list datasets <pattern>
ami show dataset info <dataset_name>
ami show tag info <tag_name>
```

#### 2. Rucio Management (`rucio-management/`)
**File**: `rucio-management/SKILL.md`

**Content structure**:
- **YAML frontmatter**: Name, description
- **Description**: Rucio DDM overview
- **When to Use**: Download datasets, manage replicas, preserve data
- **Prerequisites**: LXPlus, VOMS proxy, storage quota
- **How to Use**: Step-by-step instructions
  - Setup Rucio environment
  - Find dataset names
  - List dataset contents
  - Download datasets
  - Check data locations
  - Create replication rules (preservation)
- **Best Practices**: Data lifecycle, quota management, efficient downloads
- **Common Workflows**: 
  - Download grid job output
  - Download MC datasets
  - Preserve analysis output
- **Integration**: Rucio + AMI, Rucio + PanDA, Rucio + Analysis Frameworks
- **Resources**: Rucio docs, ATLAS DDM twiki
- **Common Issues**: Authentication, downloads, quota

**Key commands covered**:
```bash
setupATLAS
localSetupRucioClients
voms-proxy-init -voms atlas
rucio -v download <dataset_name>
rucio list-files <dataset_name>
rucio list-dataset-replicas <dataset_name>
rucio add-rule <dataset_name> 1 <RSE>
```

#### 3. ATLAS Run Query (`atlas-runquery/`)
**File**: `atlas-runquery/SKILL.md`

**Content structure**:
- **YAML frontmatter**: Name, description
- **Description**: Run query overview
- **When to Use**: Check data quality, query luminosity, create GRLs
- **Prerequisites**: LXPlus, VOMS proxy, ATLAS account
- **Available Tools**: Web interface, AMI, GRLs
- **How to Use**: Step-by-step instructions
  - Access run query web interface
  - Filter runs by criteria
  - Query specific run information
  - Generate Good Runs Lists
  - Check luminosity information
- **Best Practices**: Data quality flags, run selection, trigger queries
- **Common Workflows**:
  - Find good runs for analysis
  - Check specific run details
  - Create GRL for specific trigger
- **Integration**: Run Query + AMI, + Analysis Frameworks, + Luminosity
- **Resources**: Run query web, GRL twiki, data quality
- **Common Issues**: Access, GRL files, luminosity

**Key resources**:
- Web: https://atlas-runquery.cern.ch/
- GRLs: https://twiki.cern.ch/twiki/bin/view/AtlasProtected/GoodRunLists

### Documentation

Created comprehensive documentation:

#### ATLAS_SKILLS.md
**File**: `libs/chatlas-agents/skills/ATLAS_SKILLS.md`

**Content**:
- Overview of all three ATLAS skills
- Quick reference for each skill
- Installation instructions (user vs project skills)
- Common workflows combining multiple tools
- Prerequisites summary
- Skill development guide
- Additional resources

#### README.md Updates
Updated main README to:
- Add ATLAS Software Tools Skills section
- Link to individual skill documentation
- Mark TODO item as complete
- Explain skills are designed for LXPlus with CVMFS

### Testing

Created validation script to verify:
- ✅ All SKILL.md files have valid YAML frontmatter
- ✅ Required fields (name, description) are present
- ✅ Frontmatter parsing works correctly

Test results: All three ATLAS skills parse successfully.

## Skills Design Principles

### 1. Progressive Disclosure
- Concise YAML description for discovery
- Detailed markdown content for execution
- Agent loads skill only when needed

### 2. Self-Documenting
- Complete step-by-step instructions
- Examples for common use cases
- Troubleshooting section
- External resource links

### 3. LXPlus Native
- Assumes ATLAS software stack via CVMFS
- No installation required
- Commands work out-of-the-box on LXPlus
- VOMS proxy authentication pattern

### 4. Workflow Integration
- Skills work together (AMI → Rucio workflow)
- Integration with ATLAS tools (PanDA, Analysis frameworks)
- Real-world analysis scenarios

### 5. Best Practices
- Data lifecycle management
- Quota considerations
- Authentication patterns
- Error handling

## Usage Pattern

When an agent with these skills receives a task:

1. **Discovery**: Agent sees skill names/descriptions in system prompt
   ```
   Available Skills:
   - ami-query: Query ATLAS Metadata Interface for dataset information...
   - rucio-management: Manage ATLAS grid data with Rucio...
   - atlas-runquery: Query ATLAS run and data quality information...
   ```

2. **Recognition**: Agent identifies relevant skill for task
   ```
   User: "Download the DAOD_PHYS dataset for Higgs to ZZ analysis"
   Agent: This requires both AMI (to find dataset) and Rucio (to download)
   ```

3. **Loading**: Agent reads full SKILL.md files
   ```
   Agent: read_file ~/.deepagents/agent/skills/ami-query/SKILL.md
   Agent: read_file ~/.deepagents/agent/skills/rucio-management/SKILL.md
   ```

4. **Execution**: Agent follows skill instructions
   ```bash
   # From ami-query skill
   setupATLAS
   voms-proxy-init -voms atlas
   lsetup pyami
   ami list datasets mc20_13TeV.%.Higgs_ZZ%.DAOD_PHYS.%
   
   # From rucio-management skill
   localSetupRucioClients
   rucio -v download mc20_13TeV:<selected_dataset>
   ```

## Benefits

### For ATLAS Users
- **Structured guidance** for complex ATLAS tools
- **Best practices** embedded in skills
- **Workflow integration** between tools
- **Troubleshooting** help included

### For AI Agents
- **Discoverability** via skill names/descriptions
- **Context-appropriate** loading (progressive disclosure)
- **Step-by-step** execution instructions
- **Consistent patterns** across tools

### For ChATLAS
- **No code changes** required (pure documentation)
- **Easy to extend** with new ATLAS tools
- **Maintainable** as markdown files
- **Version controllable** in git

## Future Enhancements

### Additional ATLAS Skills
Potential future skills:
- **PanDA job submission** - Submit and monitor grid jobs
- **Athena analysis** - Run ATLAS analysis framework
- **Indico meetings** - Query upcoming ATLAS meetings
- **Physics validation** - Standard validation procedures
- **GRL creation** - Advanced Good Runs List generation

### MCP Server Implementation
Long-term: Create proper MCP server with native ATLAS tools:
- Direct Python API integration (not shell commands)
- Better error handling and validation
- Streaming results for large queries
- Cross-platform compatibility (not just LXPlus)

## Conclusion

Successfully implemented ATLAS software tools as DeepAgents skills using:
- Research via ChATLAS MCP (`search_chatlas`) for ATLAS tool documentation
- Research via LangChain docs MCP for skills architecture
- Comprehensive SKILL.md files following progressive disclosure pattern
- Extensive documentation and usage examples
- Validation testing for YAML frontmatter parsing

The skills provide agents with structured, discoverable guidance for working with ATLAS data management tools on LXPlus, enabling complex workflows like dataset discovery, download, and analysis preparation.

## Files Created

```
libs/chatlas-agents/skills/
├── ATLAS_SKILLS.md                           # Overview documentation
├── ami-query/
│   └── SKILL.md                               # AMI query skill
├── rucio-management/
│   └── SKILL.md                               # Rucio management skill
└── atlas-runquery/
    └── SKILL.md                               # Run query skill
```

Updated:
- `README.md` - Added ATLAS skills section and updated TODO

Total: 4 new files, 1 updated file
