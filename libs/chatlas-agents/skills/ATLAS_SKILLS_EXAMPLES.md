# Using ATLAS Skills - Example Scenarios

This document demonstrates how agents would use the ATLAS skills in practice.

## Scenario 1: Finding and Downloading a Dataset

**User Request:**
> "I need to download the DAOD_PHYS dataset for mc20 Higgs to ZZ analysis"

**Agent Workflow:**

1. **Skill Discovery** (from system prompt):
   - Sees "ami-query: Query ATLAS Metadata Interface for dataset information..."
   - Sees "rucio-management: Manage ATLAS grid data with Rucio..."

2. **Skill Loading**:
   ```
   Agent reads: ~/.deepagents/agent/skills/ami-query/SKILL.md
   Agent reads: ~/.deepagents/agent/skills/rucio-management/SKILL.md
   ```

3. **Execution** (following skills):
   ```bash
   # From ami-query skill instructions
   # Note: User must have already run setupATLAS, lsetup pyami, and voms-proxy-init before starting agent
   ami list datasets mc20_13TeV.%.Higgs*ZZ%.DAOD_PHYS.%
   
   # User selects specific dataset
   ami show dataset info mc20_13TeV.602074.PhPy8EG_PDF4LHC21_gg1200NW_ZZ_4lep.deriv.DAOD_PHYS.e8530_s3797_r13167_p6490
   
   # From rucio-management skill instructions
   # Note: User must have already run localSetupRucioClients
   rucio list-dataset-replicas mc20_13TeV:mc20_13TeV.602074.PhPy8EG.DAOD_PHYS.p6490
   rucio -v download mc20_13TeV:mc20_13TeV.602074.PhPy8EG.DAOD_PHYS.p6490
   ```

**Result:** Dataset successfully downloaded to local directory.

---

## Scenario 2: Retrieving Grid Job Output

**User Request:**
> "My PanDA job finished. Can you download the output files and save them permanently?"

**Agent Workflow:**

1. **Skill Discovery**:
   - Recognizes "rucio-management" skill applies

2. **Skill Loading**:
   ```
   Agent reads: ~/.deepagents/agent/skills/rucio-management/SKILL.md
   ```

3. **Interaction**:
   ```
   Agent: "I'll help you download and preserve your PanDA output. First, I need the container name from BigPanDA."
   Agent: "Please go to https://bigpanda.cern.ch/ → My BigPanDA → Your Task → Containers section"
   Agent: "What's the output container name? (Should end with _output.h5 or similar)"
   
   User: "user.jsmith:user.jsmith.12345678._000001.output_h5"
   ```

4. **Execution** (following rucio-management skill):
   ```bash
   # User must have already run setupATLAS, localSetupRucioClients, and voms-proxy-init
   
   # Check what's in the output
   rucio list-files user.jsmith:user.jsmith.12345678._000001.output_h5
   
   # Download output
   rucio -v download user.jsmith:user.jsmith.12345678._000001.output_h5
   
   # Create preservation rule (grid scratch files deleted after ~2 weeks)
   rucio add-rule user.jsmith:user.jsmith.12345678._000001.output_h5 1 CERN-PROD_USERDISK
   
   # Verify rule creation
   rucio list-rules user.jsmith:user.jsmith.12345678._000001.output_h5
   ```

**Result:** Files downloaded locally and permanently preserved on CERN-PROD_USERDISK.

---

## Scenario 3: Creating Good Runs List for Analysis

**User Request:**
> "I need a Good Runs List for 2023 data with the single muon trigger for my top quark analysis"

**Agent Workflow:**

1. **Skill Discovery**:
   - Recognizes "atlas-runquery" skill applies

2. **Skill Loading**:
   ```
   Agent reads: ~/.deepagents/agent/skills/atlas-runquery/SKILL.md
   ```

3. **Guidance** (from atlas-runquery skill):
   ```
   Agent: "I'll help you create a GRL for 2023 data with the single muon trigger. Here's what to do:"
   
   1. Go to https://atlas-runquery.cern.ch/
   2. Apply these filters:
      - Year: 2023
      - Stream: physics_Main
      - Trigger: HLT_mu26_ivarmedium (or your specific trigger)
      - DQ: PHYS_StandardGRL_All_Good
   3. Verify the trigger was unprescaled (prescale = 1)
   4. Export the matching runs as XML format
   5. Save the GRL file to your analysis area
   
   Agent: "Alternatively, you can use the official GRL for 2023:"
   Agent: "Check https://twiki.cern.ch/twiki/bin/view/AtlasProtected/GoodRunLists"
   ```

4. **Follow-up** (if using official GRL):
   ```bash
   # Download official GRL
   wget https://atlas-runquery.cern.ch/GRL/.../data23_13p6TeV.periodAllYear_PHYS_StandardGRL_All_Good.xml
   
   # Use in AnalysisTop
   # Add to TopConfig:
   # GRLDir: GoodRunsLists/
   # GRLFile: data23_13p6TeV.periodAllYear_PHYS_StandardGRL_All_Good.xml
   ```

**Result:** User has appropriate GRL for their 2023 top quark analysis.

---

## Scenario 4: Multi-Tool Workflow

**User Request:**
> "Find all available Run 3 Z to ee datasets from 2023, check the luminosity, and download them"

**Agent Workflow:**

1. **Skill Discovery**:
   - Uses ami-query, atlas-runquery, and rucio-management together

2. **Execution**:
   
   **Step 1 - Find datasets (ami-query)**:
   ```bash
   # User must have already run setupATLAS, lsetup pyami, and voms-proxy-init before starting agent
   ami list datasets data23_13p6TeV.%.Zee%.DAOD_PHYS.%
   ```

   **Step 2 - Check runs and luminosity (atlas-runquery)**:
   ```
   Agent: "Go to https://atlas-runquery.cern.ch/"
   Agent: "Filter by Year: 2023, Stream: physics_Main, DQ: PHYS_StandardGRL_All_Good"
   Agent: "The integrated luminosity for 2023 data is shown in the results"
   ```

   **Step 3 - Download datasets (rucio-management)**:
   ```bash
   # User must have already run localSetupRucioClients
   
   # Check size first
   rucio list-dataset-replicas data23_13p6TeV:<selected_dataset>
   
   # Download
   rucio download --dir $HOME/analysis/zee_data data23_13p6TeV:<selected_dataset>
   ```

**Result:** User has Z→ee datasets from 2023 downloaded with knowledge of the luminosity.

---

## How Skills Work Together

### Progressive Disclosure

**Initial State** (agent sees in system prompt):
```
Available Skills:

User Skills:
- ami-query: Query ATLAS Metadata Interface (AMI) for dataset information...
  → Read ~/.deepagents/agent/skills/ami-query/SKILL.md for full instructions

- rucio-management: Manage ATLAS grid data and samples with Rucio...
  → Read ~/.deepagents/agent/skills/rucio-management/SKILL.md for full instructions

- atlas-runquery: Query ATLAS run and data quality information...
  → Read ~/.deepagents/agent/skills/atlas-runquery/SKILL.md for full instructions
```

**When Task Arrives**:
1. Agent matches task to skill(s) based on descriptions
2. Agent reads relevant SKILL.md file(s)
3. Agent follows step-by-step instructions
4. Agent executes commands as documented in skills

### Skills Complement Each Other

**AMI + Rucio**:
- AMI finds datasets → Rucio downloads them

**Run Query + AMI**:
- Run Query creates GRL → AMI finds datasets for those runs

**All Three Together**:
- Run Query selects good runs → AMI finds datasets → Rucio downloads data

## Benefits for Users

1. **Guided Workflows**: Step-by-step instructions embedded in skills
2. **Best Practices**: Skills include ATLAS community standards
3. **Error Recovery**: Troubleshooting sections help with common issues
4. **Resource Links**: Direct links to official ATLAS documentation
5. **Examples**: Real-world scenarios demonstrate usage patterns

## Benefits for Agents

1. **Discoverability**: Skill names/descriptions in system prompt
2. **Context**: Full instructions loaded when needed (not always in context)
3. **Consistency**: Same pattern across all ATLAS tools
4. **Maintainability**: Skills are pure markdown, easy to update

## Installation

Users can install these skills in two ways:

**Important:** Before starting the agent, users must initialize their ATLAS environment:
```bash
setupATLAS
lsetup pyami              # For AMI queries
localSetupRucioClients    # For Rucio data management
voms-proxy-init -voms atlas
```

**Note:** Not all commands are needed for all skills. See individual skill prerequisites for details.

### Option 1: User Skills (per-agent)
```bash
cp -r libs/chatlas-agents/examples/skills/ami-query ~/.deepagents/agent/skills/
cp -r libs/chatlas-agents/examples/skills/rucio-management ~/.deepagents/agent/skills/
cp -r libs/chatlas-agents/examples/skills/atlas-runquery ~/.deepagents/agent/skills/
```

### Option 2: Project Skills (per-project)
```bash
mkdir -p .deepagents/skills
cp -r libs/chatlas-agents/examples/skills/ami-query .deepagents/skills/
cp -r libs/chatlas-agents/examples/skills/rucio-management .deepagents/skills/
cp -r libs/chatlas-agents/examples/skills/atlas-runquery .deepagents/skills/
```

## Verification
Check skills are available:
```bash
deepagents skills list
```

Expected output:
```
Available Skills:

User Skills:
  • ami-query
    Query ATLAS Metadata Interface (AMI) for dataset information...
    Location: ~/.deepagents/agent/skills/ami-query/

  • rucio-management
    Manage ATLAS grid data and samples with Rucio...
    Location: ~/.deepagents/agent/skills/rucio-management/

  • atlas-runquery
    Query ATLAS run and data quality information...
    Location: ~/.deepagents/agent/skills/atlas-runquery/
```
