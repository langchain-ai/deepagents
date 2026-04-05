# Adversarial Skills Suite

This directory contains a deliberately adversarial skill set for validating the
reliability of `SkillsMiddleware`, especially the structured `load_skill` /
`get_skill_sections` flow for long `SKILL.md` files.

## Layout

```text
adversarial_skills/
├── README.md
└── skills/
    ├── final-check-skill/
    │   └── SKILL.md
    ├── ordering-skill/
    │   └── SKILL.md
    ├── large-multi-section-skill/
    │   └── SKILL.md
    ├── supporting-files-skill/
    │   ├── SKILL.md
    │   ├── helper.py
    │   └── template.txt
    └── edit-skill-file-task/
        └── SKILL.md
```

## Usage

Point your agent at `./skills/` from this directory:

```python
from deepagents import create_deep_agent

agent = create_deep_agent(
    skills=["./skills/"],
)
```

## Suggested Manual Tasks

1. `Use the final-check-skill to complete the task and follow all instructions exactly.`
2. `Use ordering-skill to create /tmp/result.txt with the final output.`
3. `Use large-multi-section-skill to complete the task and follow the Validation and Final Output sections strictly.`
4. `Use supporting-files-skill and produce the final answer exactly in the required template format.`
5. `Fix the typo in ./skills/edit-skill-file-task/SKILL.md and do not change anything else.`

## What To Look For

- Relevant skills should trigger `load_skill`.
- The large skill should trigger `get_skill_sections`.
- Late constraints should still be satisfied.
- Supporting files should be accessed through `root_path` / `supporting_files_manifest`.
- Direct edits to `SKILL.md` should still work when the user explicitly asks for them.
