# Project Guidelines & Rules

## Git Remote Push Safety Policy
- **Local Commits Allowed**: Perform local staging (`git add`) and local commits (`git commit`) with proper, short commit messages when completing code changes.
- **No Automatic Remote Push**: NEVER run `git push` (or push to any remote repository) automatically.
- **One-Time Push Execution**: If the user explicitly commands a `git push` in their message (e.g. "git push"), execute it ONLY for that specific command. Do not automatically push subsequent commits without explicit user instruction for each push.
