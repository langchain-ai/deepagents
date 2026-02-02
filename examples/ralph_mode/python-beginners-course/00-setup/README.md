# Module 0: Setup + Terminal + Git

## Objectives
- Install Python and verify it works
- Create and activate a virtual environment
- Use the terminal to navigate and run scripts
- Use Git to track your work

## 1) Install Python
Recommended: Python 3.11+.

Verify:
```bash
python3 --version
python3 -c "print('hello')"
```

## 2) Create a course workspace
```bash
mkdir -p ~/code/python-course
cd ~/code/python-course
```

## 3) Virtual environment
```bash
python3 -m venv .venv
source .venv/bin/activate
python -V
```

## 4) Editor
Use VS Code (recommended) and install the Python extension.

## 5) Terminal basics
- `pwd`, `ls`, `cd`
- `mkdir`, `touch`
- `cat`, `code .`

## 6) Git basics (local)
Initialize a repo and make your first commits.

```bash
git init
printf "print('Hello, Python!')\n" > hello.py
python hello.py

git status
git add hello.py
git commit -m "Add hello world"
```

## 7) Optional: GitHub
If using GitHub:
- create an empty repo
- add remote: `git remote add origin <URL>`
- push: `git push -u origin main` (or `master`)

## Checkpoint
- You can run `python` inside `.venv`
- You can make a commit
