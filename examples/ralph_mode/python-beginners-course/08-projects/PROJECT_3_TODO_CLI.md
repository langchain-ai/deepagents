# Project 3: To-do CLI (JSON persistence)

## Goal
A simple to-do list manager that saves to disk.

## Requirements
- Commands:
  - `add <task>`
  - `list`
  - `done <id>`
- Store tasks in `tasks.json`

## Data format
A list of dicts, each:
- `id` (int)
- `text` (str)
- `done` (bool)

## Git
- Branch: `project/03-todo`
- Suggested commits:
  - "Add task model and load/save"
  - "Implement add/list"
  - "Implement done"
