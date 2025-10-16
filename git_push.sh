#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 \"commit message\""
  exit 1
fi

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$REPO_ROOT"

echo "Staging all changes..."
git add -A

echo "Creating commit..."
git commit -m "$*"

echo "Pulling latest changes with rebase..."
git pull --rebase origin main

echo "Pushing to origin/main..."
git push origin main

echo "Done."
