# MimicDroid / DexMimicGen — work context & locations

Reference notes for prior work on MimicDroid data + DexMimicGen (MimicGen) data
generation. Written 2026-06-23. Paths are on the shared Lustre mount `/mnt/ddn`
unless marked **/tmp (ephemeral — may vanish on reboot)**.

## MimicDroid

- **Code repo (this repo)**: `/mnt/ddn/mimicdroid-robocasa`
  - Git: fork of `github.com/UT-Austin-RPL/mimicdroid-robocasa` (robocasa-based).
  - Key files:
    - `robocasa/scripts/download_mimicdroid_dataset.py` — dataset downloader
    - `robocasa/utils/mimicdroid_utils.py` — utilities
    - `DATASET.md`, `GETTING_STARTED.md`, `docs/`
  - `robocasa/` here is also the **editable robocasa 0.2.0** source used by the
    dexmimicgen env below.
- **Dataset**: `/mnt/ddn/dataset/MimicDroidDataset`
  - `GR1/`, `TaskDemos/`, `training/`, `README.md`, `_download.log`
- **Task list**: `/tmp/mimicdroid_tasks.json` **/tmp**
- **Related memory note** (auto-memory):
  `~/.claude/projects/-mnt-ddn-EgoVerse/memory/egoverse-mimicdroid-curation.md`
  — egoverse local set was curated to align with MimicDroid PnP tasks
  (PnPSink→Counter/Cabinet/Microwave, TurnOnFaucet, CloseCabinetDoor). aria/eva
  have only pick-and-place overlap (no faucet/cabinet/sink).

## DexMimicGen / MimicGen

- **Env (venv, py3.10)**: `/mnt/ddn/dexmimicgen_env`
  - Editable installs: `mimicgen 1.0.0`, `robomimic 0.3.0`, `robosuite 1.5.0`,
    `robocasa 0.2.0`.
- **Source clones** (where the editable installs point):
  - robosuite → `/mnt/ddn/robosuite/robosuite`
  - robocasa  → `/mnt/ddn/mimicdroid-robocasa/robocasa` (shared with this repo)
  - mimicgen  → `/tmp/mimicgen_robocasa/mimicgen` **/tmp**
  - robomimic → `/tmp/robomimic_src` **/tmp**
- **Investigation / data-gen work**: `/mnt/ddn/dexmimicgen_env/scratch/`
  - `turnon_faucet_mg.json` — MimicGen config (task `TurnOnFaucet`,
    env `TurnOnSinkFaucet`, subtask term signal `stage_contact_handle`)
  - `TurnOnFaucet/demo.hdf5` — source demos
  - `gen/turnon_faucet_pilot/` — a MimicGen generation pilot run
    (`mg_config.json`, `log.txt`, `logs/`, `tmp/`, `tmp_failed/`)
  - **STATUS: pilot FAILED** — robosuite raised
    `AssertionError: environment got invalid action dimension -- expected 14,
    got 12` (`/mnt/ddn/robosuite/.../robot_env.py:583` in `_pre_action`). I.e. a
    bimanual GR1 action-dim mismatch (env wants 14-D, generator fed 12-D). This
    is the open blocker to resolve before DexMimicGen generation works here.

## Caveats
- mimicgen + robomimic sources live under **/tmp** — copy to `/mnt/ddn` before
  relying on them long-term (a reboot can wipe /tmp).
- Running the gen env: `/mnt/ddn/dexmimicgen_env/bin/python` (robosuite/robocasa
  need EGL/MuJoCo libs for rendering).
