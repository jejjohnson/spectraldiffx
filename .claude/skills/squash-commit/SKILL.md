---
name: squash-commit
description: Write a Conventional-Commits squash commit message for a spectraldiffx pull request from its commits and description. Use when asked for a squash / merge commit message.
---

Generate a squash commit message for a GitHub PR.

## Instructions

1. If an argument is provided (PR number or URL), fetch the PR details
   (`gh pr view`, or the GitHub MCP tools). Otherwise, detect the current
   branch and find its open PR.
2. Fetch all individual commit messages in the PR (`gh api` or `git log`).
3. Combine them into a single squash commit message following these rules.

### Format

```
<type>(<scope>): <concise summary of the overall change>

<1-3 sentence description combining the key changes from all commits.
Focus on the "why" and overall effect, not per-commit details.>

Co-authored-by: <preserve all unique Co-authored-by lines from the individual commits>
```

### Rules

- Conventional Commits types: `feat`, `fix`, `docs`, `style`, `refactor`,
  `perf`, `test`, `build`, `ci`, `chore`, `revert`; a lowercase subject (CI
  validates PR titles against `^[a-z].*$`). The scope is the area the
  changelog uses: `fourier`, `transforms`, `solvers`, `capacitance`,
  `operators`, `filters`, `grid`, `chebyshev`, `spherical`, `docs`, `deps`,
  `agents`, ….
- release-please builds `CHANGELOG.md` from the squash commits on `main`:
  `feat`, `fix`, `perf` and `revert` are listed (Features, Bug Fixes,
  Performance Improvements, Reverts); `docs`, `style`, `refactor`, `test`,
  `build`, `ci` and `chore` are hidden. So the type decides whether a change
  appears in the release notes. Releases are plain semver tags (`0.1.1`, no
  `v`), and before 1.0 a `feat` bumps the patch version and a breaking
  change the minor (`bump-patch-for-minor-pre-major`,
  `bump-minor-pre-major` in `release-please-config.json`).
- The type/scope reflect the dominant change across all commits; mention the
  rest in the body.
- A breaking change keeps its `!` and `BREAKING CHANGE:` footer (and says
  what finitevolX must change, if it imports the name).
- Reference the issue the PR closes (`closes #NN`) in the body when there is
  one.
- Collapse redundant or incremental commits into one coherent description.
- Preserve ALL unique `Co-authored-by` lines.
- Keep the summary line under 72 characters.
- Do NOT list the individual commit messages as bullet points — this is a
  squash, not a merge.

### Output

Print ONLY the final squash commit message in a code block so the user can
copy it directly. Do not add explanation before or after.
