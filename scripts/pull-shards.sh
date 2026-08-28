#!/usr/bin/env bash
# Laptop-side watcher: repeatedly rsync any newly-saved MIMO shards down from the box, so a pod
# restart/expiry can never wipe completed data. Run on the LAPTOP while gen-mimo-safe.sh runs on box.
#   bash scripts/pull-shards.sh [REMOTE_DIR] [LOCAL_DIR] [MAX_MIN]
# NOTE: deliberately NOT using `set -e` — rsync/ls return non-zero before any shard exists, which
# must not kill the watcher.
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/greenland-config.sh   # gives LOCAL_PORT, SSH_USER

REMOTE=${1:-'~/World-Model-Channel-Estimation/data/mimo'}
LOCAL=${2:-data/mimo}
MAX_MIN=${3:-90}
mkdir -p "$LOCAL"
SSH="ssh -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=20 -p $LOCAL_PORT"

echo ">> watching $REMOTE -> $LOCAL (pull complete .pt shards every 60s, up to ${MAX_MIN}m)"
for i in $(seq 1 $((MAX_MIN))); do
  rsync -avz --ignore-existing -e "$SSH" \
    "$SSH_USER@localhost:$REMOTE/shard_*.pt" "$LOCAL/" 2>/dev/null || true
  n=$(ls "$LOCAL"/shard_*.pt 2>/dev/null | wc -l | tr -d ' ')
  echo "  [$i min] local shards: $n/8"
  [ "$n" -ge 8 ] && { echo ">> all 8 shards pulled"; break; }
  sleep 60
done
