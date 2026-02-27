#!/usr/bin/env bash
set -euo pipefail

TAG_NAME="freeze-v0.9-sse-pricefix-1"
TAG_MSG="SommelierAI v0.9 freeze: stable stream+price sort+scroll+color guard (clean+gitignore)"
CLEANUP_COMMIT_MSG="CLEANUP: add gitignore + untrack generated files"

REMOTE_NAME="origin"
REMOTE_URL="${1:-}"

echo "== SommelierAI Freeze Cleanup =="

git rev-parse --is-inside-work-tree >/dev/null 2>&1 || { echo "Not a git repo."; exit 1; }

BRANCH="$(git rev-parse --abbrev-ref HEAD)"
echo "On branch: $BRANCH"

cat > .gitignore <<'EOG'
.DS_Store
**/.DS_Store
__pycache__/
**/__pycache__/
*.pyc
*.pyo
*.pyd
*.log
out.json
data/out.json
backend/out.json
backend/main_A9_*.py
backend/mainSAFE*.py
EOG

git rm -r --cached --ignore-unmatch .DS_Store backend/.DS_Store "app docs/.DS_Store" >/dev/null 2>&1 || true
git rm -r --cached --ignore-unmatch backend/__pycache__ >/dev/null 2>&1 || true
git rm --cached --ignore-unmatch backend/out.json data/out.json >/dev/null 2>&1 || true
git rm --cached --ignore-unmatch "backend/1{print" >/dev/null 2>&1 || true

git rm --cached --ignore-unmatch backend/main_A9_1_fixed.py backend/main_A9_2.py backend/main_A9_3.py backend/main_A9_4_FIXED.py backend/main_A9_4_FIXED2.py >/dev/null 2>&1 || true
git rm --cached --ignore-unmatch "backend/mainSAFE (A7 + piccoli A10).py" backend/main_FREEZE_FINAL.py backend/main_SAFE_PERF.py >/dev/null 2>&1 || true

git add .gitignore || true

if git diff --cached --quiet; then
  echo "No staged changes to commit."
else
  git commit -m "$CLEANUP_COMMIT_MSG"
  echo "Cleanup commit created."
fi

if git rev-parse "$TAG_NAME" >/dev/null 2>&1; then
  git tag -d "$TAG_NAME" >/dev/null 2>&1 || true
fi
git tag -a "$TAG_NAME" -m "$TAG_MSG"
echo "Tag set: $TAG_NAME -> $(git rev-parse HEAD)"

echo "Verifying tracked junk..."
if git ls-tree -r --name-only HEAD | egrep -i '\.DS_Store|__pycache__|out\.json|backend/main_A9_|backend/mainSAFE|backend/1\{print' >/dev/null; then
  echo "WARNING: Some junk is still tracked."
  git ls-tree -r --name-only HEAD | egrep -i '\.DS_Store|__pycache__|out\.json|backend/main_A9_|backend/mainSAFE|backend/1\{print' || true
  exit 2
else
  echo "OK: no junk tracked."
fi

if [[ -n "$REMOTE_URL" ]]; then
  if git remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    echo "Remote exists: $(git remote get-url "$REMOTE_NAME")"
  else
    git remote add "$REMOTE_NAME" "$REMOTE_URL"
    echo "Remote added: $REMOTE_URL"
  fi

  git push -u "$REMOTE_NAME" "$BRANCH"
  git push -f "$REMOTE_NAME" "$TAG_NAME"
  echo "Push complete."
else
  echo "No remote URL provided. Skipping push."
fi

echo "== DONE =="
