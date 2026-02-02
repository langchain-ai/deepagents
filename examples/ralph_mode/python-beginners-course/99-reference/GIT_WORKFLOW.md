# Git workflow for this course

## Minimum commands you need
- `git status`
- `git add -A`
- `git commit -m "message"`
- `git log --oneline --decorate --graph --all`
- `git switch -c <branch>` (create branch)
- `git switch <branch>` (switch)

## Recommended routine (per exercise)
1. Create a branch
   - `git switch -c exercise/01-basics-variables`
2. Work in small steps
3. Commit small steps
   - `git add -A`
   - `git commit -m "Solve variables warmup"`
4. Merge back (local-only option)
   - `git switch main` (or `master`)
   - `git merge exercise/01-basics-variables`

## Optional GitHub routine
- Push branch: `git push -u origin exercise/01-basics-variables`
- Open a PR
- Get review feedback
- Merge PR

## Commit message tips
Good:
- "Add unit tests for is_prime"
- "Handle empty input in text analyzer"

Avoid:
- "stuff"
- "fix"
