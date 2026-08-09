---
name: git-push-policy
description: >-
  Strict git remote push safety policy for ComfyUI-llama-cpp_vlm project.
  Use when performing git operations or committing changes.
---

# Git Push Safety Policy

This skill enforces strict safety protocols around git operations for this project.

## Core Directives

1. **Local Commits Only**:
   - Perform local staging (`git add`) and local commits (`git commit`) with a proper, concise commit message when completing work or asked to commit.

2. **No Automatic Remote Push**:
   - **NEVER** run `git push` (or push to any remote repository) automatically under any circumstances.

3. **One-Time Push Authorization**:
   - If the user explicitly asks to `git push` in a prompt, execute the push **only for that single request**.
   - Do NOT treat a previous push request as permission or an instruction for subsequent turns or future commits.
