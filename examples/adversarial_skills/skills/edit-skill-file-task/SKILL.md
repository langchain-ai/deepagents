---
name: edit-skill-file-task
description: A simple skill file used to verify that direct SKILL.md editing is not confused with skill activation.
---

# Edit Skill File Task

## Purpose

This file exists so a user can explicitly request modifications to a `SKILL.md`
file and verify that the middleware does not incorrectly force the execution
path through managed skill loading.

## Current Notes

- There is an intentional typo in this section: valdiation.
- Tests may ask the agent to fix that typo directly in this file.
- Direct edits should preserve all other content.

## Reminder

When the user explicitly asks to modify this file, treat it as a file editing
task, not as a skill execution task.
