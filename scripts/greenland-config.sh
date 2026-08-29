#!/usr/bin/env bash
#
# greenland-config.sh — shared config for all Greenland scripts.
# ---------------------------------------------------------------
# Sourced by greenland-auth.sh, greenland-connect.sh, greenland-sync.sh.
# To switch instances, edit ONLY the INSTANCE-SPECIFIC block below
# (copy the values from the Greenland Console job JSON).

# ---- Account / role (stable across instances for this initiative) ----
ACCOUNT="703671891219"
CUSTOMER_ROLE="Intern"
PROVIDER="isengard"                 # conduit is denied for this alias; isengard works
PROFILE="greenland-exp3"   # NOT "greenland" — that profile drifts to the wrong account and every
                           # StartSession returns a misleading TargetNotConnected. exp3 = acct 072510399842.
REGION="us-east-2"
JOB_ROLE_ARN="arn:aws:iam::072510399842:role/greenland-access-37f871283e3e69fdbfe97939a34079a8bfdfdd85"

# ---- INSTANCE-SPECIFIC (edit when you switch jobs) -------------------
# Job: cmohsinm-workspace (EKS: cmohsinm-workspace-b05ab0b7), JobId d208ff94-...
# 1x p4d.24xlarge, 8x A100, us-east-2. Initiative: KiroScienceInterns.
SSM_TARGET="mi-0262e3346abe3a117"   # SsmManagedInstanceId from job JSON
MAIN_NODE_IP="10.3.44.81"           # MainNodeIP / NodesEniHostIP
LOCAL_PORT="1055"                   # local SSH tunnel port (per-instance to avoid clashes)
# ----------------------------------------------------------------------

# ---- SSH / tunnel (stable) ----
REMOTE_PORT="2222"                  # sshd port inside the pod
SSH_USER="greenland-user"

# ---- Sync (rsync over the tunnel) ----
# Local project root -> remote workspace dir.
LOCAL_PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
REMOTE_PROJECT_DIR="/home/greenland-user/World-Model-Channel-Estimation"
