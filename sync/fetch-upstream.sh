#!/bin/bash
# Fetch and check out a specific upstream pi tag into upstream/pi.
set -euo pipefail
TAG="${1:?usage: fetch-upstream.sh <tag>}"
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
git -C "$ROOT/upstream/pi" fetch --quiet origin tag "$TAG" --no-tags
git -C "$ROOT/upstream/pi" checkout --quiet "$TAG"
echo "upstream checked out at $TAG"
