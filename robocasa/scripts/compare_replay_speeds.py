#!/usr/bin/env python3
"""
Compare dataset-replay success rates across playback speeds (e.g. 1.0x vs 0.7x).

For every episode in the given datasets, this replays the recorded absolute
actions at each requested speed and records whether ``env._check_success()``
passes. It then prints a per-task and overall success-rate comparison table.

Slowing playback down (speed < 1.0) inserts interpolated absolute targets between
the recorded ones (see ``resample_actions`` in robocasa.utils.mimicdroid_utils),
which gives the IK controller more time to track each target and typically
reduces replay failures.

Example
-------
    MUJOCO_GL=egl uv run python robocasa/scripts/compare_replay_speeds.py \
        --dataset_root /mnt/ddn/dataset/MimicDroidDataset/TaskDemos \
        --robots GR1TwoHand --speeds 1.0 0.7
"""

import os
import sys
import glob
import json
import argparse

import h5py
import numpy as np
from termcolor import colored

# Match replay_dataset.py: make the icrt directory importable if present.
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../icrt"))

import robocasa  # noqa: F401  (registers envs)
import robocasa.macros as macros
from robocasa.environments.kitchen.play_env.play_env import *  # noqa: F401,F403

from robocasa.utils.mimicdroid_utils import (
    make_env,
    reset_to,
    resample_actions,
    EnvArgs,
)

macros.SHOW_SITES = False


def discover_datasets(args):
    """Return a sorted list of (task_name, hdf5_path) to evaluate."""
    if args.hdf5_paths:
        paths = list(args.hdf5_paths)
    else:
        pattern = os.path.join(args.dataset_root, "**", args.hdf5_name)
        paths = glob.glob(pattern, recursive=True)
    paths = sorted(set(paths))

    datasets = []
    for p in paths:
        # Task name = the directory two levels up (".../<Task>/<id>/<file>.hdf5").
        task_name = os.path.basename(os.path.dirname(os.path.dirname(p)))
        datasets.append((task_name, p))
    if args.max_tasks is not None:
        datasets = datasets[: args.max_tasks]
    return datasets


def run_episode(env, hdf5_path, ep_name, robots, speed, render=False, max_steps=None):
    """Reset to an episode's initial state, replay at ``speed``, return success bool."""
    with h5py.File(hdf5_path, "r") as f:
        ep_group = f["data"][ep_name]
        states = ep_group["states"][:]
        actions = ep_group["actions"][:]

        ep_meta = ep_group.attrs.get("ep_meta", {})
        if isinstance(ep_meta, str):
            ep_meta = json.loads(ep_meta)
        model_file = ep_group.attrs.get("model_file", None)
        non_robot_qpos_idx = ep_group.attrs.get("non_robot_qpos_idx", None)
        qpos_state = ep_group.attrs.get("initial_qpos", None)

    initial_state = {"states": states[0], "ep_meta": ep_meta}
    if model_file is not None:
        initial_state["model"] = model_file
    if non_robot_qpos_idx is not None:
        initial_state["non_robot_qpos_idx"] = non_robot_qpos_idx
    if qpos_state is not None:
        initial_state["qpos_state"] = qpos_state

    reset_to(
        env,
        initial_state,
        replace_robot_joints=True,
        change_to_gr1="GR1" in robots,
    )

    actions = resample_actions(actions, speed)
    if max_steps is not None:
        actions = actions[:max_steps]

    for action in actions:
        if render:
            env.render()
        _, _, done, _ = env.step(action)
        if done:
            break

    return bool(env._check_success())


def _rate(cell):
    succ, tot = cell
    return (succ / tot * 100.0) if tot else 0.0


def format_table(results, speeds):
    """Render a per-task + overall table, focused on the gap vs the baseline speed.

    The first speed in ``speeds`` (e.g. 1.0x) is the baseline; every other speed
    gets an extra "Δ vs <baseline>" column showing the success-rate change in
    percentage points (positive = that speed helped).
    """
    tasks = sorted(results.keys())
    baseline = speeds[0]
    others = speeds[1:]
    name_w = max([len("Task")] + [len(t) for t in tasks])
    speed_w = 13
    delta_w = 12

    header = "Task".ljust(name_w)
    for s in speeds:
        header += "  " + f"{s}x".rjust(speed_w)
    for s in others:
        header += "  " + f"Δ({s}-{baseline})".rjust(delta_w)
    lines = [header, "-" * len(header)]

    totals = {s: [0, 0] for s in speeds}

    def row(label, cells_by_speed):
        line = label.ljust(name_w)
        for s in speeds:
            succ, tot = cells_by_speed[s]
            line += "  " + f"{succ}/{tot} ({_rate((succ, tot)):3.0f}%)".rjust(speed_w)
        base_rate = _rate(cells_by_speed[baseline])
        for s in others:
            d = _rate(cells_by_speed[s]) - base_rate
            line += "  " + f"{d:+.0f} pp".rjust(delta_w)
        return line

    for task in tasks:
        for s in speeds:
            totals[s][0] += results[task][s][0]
            totals[s][1] += results[task][s][1]
        lines.append(row(task, results[task]))

    lines.append("-" * len(header))
    lines.append(row("OVERALL", totals))
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Compare replay success rates across playback speeds"
    )
    parser.add_argument(
        "--dataset_root",
        type=str,
        default="/mnt/ddn/dataset/MimicDroidDataset/TaskDemos",
        help="Root dir searched recursively for dataset files",
    )
    parser.add_argument(
        "--hdf5_name",
        type=str,
        default="demo_im128_notp.hdf5",
        help="Dataset filename to match under dataset_root",
    )
    parser.add_argument(
        "--hdf5_paths",
        type=str,
        nargs="+",
        default=None,
        help="Explicit list of dataset files (overrides dataset_root search)",
    )
    parser.add_argument(
        "--robots",
        type=str,
        default="GR1TwoHand",
        choices=["DemoTwoHand", "GR1TwoHand"],
    )
    parser.add_argument(
        "--speeds",
        type=float,
        nargs="+",
        default=[1.0, 0.7],
        help="Playback speeds to compare (e.g. --speeds 1.0 0.7 0.5)",
    )
    parser.add_argument(
        "--max_tasks",
        type=int,
        default=None,
        help="Only evaluate the first N dataset files (for quick runs)",
    )
    parser.add_argument(
        "--max_episodes_per_file",
        type=int,
        default=None,
        help="Only evaluate the first N episodes in each file",
    )
    parser.add_argument("--render", action="store_true", help="Render during replay")
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional path to write the full results as JSON",
    )
    args = parser.parse_args()

    datasets = discover_datasets(args)
    if not datasets:
        print(colored("No dataset files found.", "red"))
        return
    print(colored(f"Found {len(datasets)} dataset file(s).", "green"))
    print(colored(f"Comparing speeds: {args.speeds}", "green"))

    # results[task][speed] = [n_success, n_total]
    results = {}

    for task_name, hdf5_path in datasets:
        print(colored(f"\n{'='*60}", "cyan"))
        print(colored(f"Task: {task_name}  ({hdf5_path})", "cyan"))
        print(colored(f"{'='*60}", "cyan"))

        env = None
        try:
            env_args = EnvArgs(
                robots=args.robots,
                render=args.render,
                control_freq=20,
                controller="WHOLE_BODY_MINK_IK" if "GR1" in args.robots else None,
                use_camera_obs=False,
            )
            env, _ = make_env(file_name=hdf5_path, env_args=env_args)

            with h5py.File(hdf5_path, "r") as f:
                ep_names = sorted(list(f["data"].keys()))
            if args.max_episodes_per_file is not None:
                ep_names = ep_names[: args.max_episodes_per_file]

            results.setdefault(task_name, {s: [0, 0] for s in args.speeds})
            for ep_name in ep_names:
                for speed in args.speeds:
                    try:
                        success = run_episode(
                            env, hdf5_path, ep_name, args.robots, speed,
                            render=args.render,
                        )
                    except Exception as e:  # noqa: BLE001
                        print(colored(f"  [{ep_name} @ {speed}x] ERROR: {e}", "red"))
                        success = False
                    results[task_name][speed][1] += 1
                    results[task_name][speed][0] += int(success)
                    color = "green" if success else "red"
                    print(
                        colored(
                            f"  {ep_name} @ {speed}x -> {'SUCCESS' if success else 'FAIL'}",
                            color,
                        )
                    )
        except Exception as e:  # noqa: BLE001
            print(colored(f"Failed to evaluate task {task_name}: {e}", "red"))
        finally:
            if env is not None:
                env.close()

        # Write results + a running table after every task so progress is
        # inspectable while a long sweep is still going.
        if args.output:
            with open(args.output, "w") as f:
                json.dump(results, f, indent=2)
        print(colored("\n[running totals]", "yellow"))
        print(format_table(results, args.speeds), flush=True)

    print(colored(f"\n{'='*60}", "magenta"))
    print(colored("SUCCESS RATE COMPARISON", "magenta", attrs=["bold"]))
    print(colored(f"{'='*60}", "magenta"))
    print(format_table(results, args.speeds))

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(colored(f"\nResults written to {args.output}", "green"))


if __name__ == "__main__":
    main()
