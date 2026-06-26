# Greenland GPU scripts

Connect to and run on the Greenland SDB (GPU box) from your laptop.

## Files
- `greenland-config.sh` — shared config. **Edit only the INSTANCE-SPECIFIC block to switch jobs.**
- `greenland-auth.sh` — daily Midway/Isengard auth + `greenland` AWS profile setup.
- `greenland-connect.sh` — open the SSM tunnel and/or SSH in.
- `greenland-sync.sh` — rsync project up/down over the tunnel.
- `remote-setup-and-test.sh` — run ON the box: builds the venv + runs GPU tests.

## Current instance (cmohsinm-workspace, EKS cmohsinm-workspace-8cd2cb8e)
- 1× p4d.24xlarge, 8× A100, us-east-2
- `SSM_TARGET=mi-0c5d9ae2bd4233d60`, `MAIN_NODE_IP=10.3.22.179`, tunnel `LOCAL_PORT=1057`

## Typical session
```bash
# Terminal 1 (laptop) — auth needs an interactive Midway PIN / security-key touch
./scripts/greenland-auth.sh
./scripts/greenland-connect.sh tunnel        # keep this open

# Terminal 2 (laptop)
./scripts/greenland-sync.sh up               # push code to the box
./scripts/greenland-connect.sh ssh           # SSH onto the box
#   on the box:
cd ~/World-Model-Channel-Estimation
bash scripts/remote-setup-and-test.sh        # venv + GPU tests

# Back on the laptop, pull figures/results:
./scripts/greenland-sync.sh down
```

## Switching instances
When you get a new job, edit the **INSTANCE-SPECIFIC** block in `greenland-config.sh`:
`SSM_TARGET` (= `SsmManagedInstanceId`), `MAIN_NODE_IP` (= `MainNodeIP`), and `LOCAL_PORT`
(I'll set whatever port you tell me). Everything else stays the same.

## Requirements (laptop)
`aws` CLI v2, `session-manager-plugin`, `mwinit`, `ada`, `rsync` — all verified present.
