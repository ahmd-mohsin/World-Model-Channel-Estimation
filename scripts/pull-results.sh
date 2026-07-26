#!/usr/bin/env bash
# Pull training artifacts from the box to local results/<tag>/ — run from the LAPTOP.
# Guards against ephemeral-box loss: call after (or during) any run.
#   bash scripts/pull-results.sh <tag>
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
source scripts/greenland-config.sh
TAG="${1:?usage: pull-results.sh <tag>}"
DST="results/$TAG"; mkdir -p "$DST"
SSH="-o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=25 -p $LOCAL_PORT"
R="greenland-user@localhost:World-Model-Channel-Estimation"
for f in "dashboard/metrics_$TAG.json" "dashboard/eval_$TAG.json" "implementation/checkpoints/sswm_e2e_$TAG.pt"; do
  scp $SSH "$R/$f" "$DST/" 2>/dev/null && echo "pulled $(basename $f)" || echo "  (not present yet: $(basename $f))"
done
ls -la "$DST/"
