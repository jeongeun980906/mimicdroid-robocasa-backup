#!/usr/bin/env python3
"""
Standalone script to replay dataset trajectories from HDF5 files.
Supports both regular robots and GR1 robots.
"""

import os
import sys
import argparse
import json
import h5py
import cv2
import numpy as np
import time
from pathlib import Path
from termcolor import colored

# Add the icrt directory to the path
sys.path.append(os.path.join(os.path.dirname(__file__), "../../../icrt"))

import robocasa
import robocasa.macros as macros
from robocasa.environments.kitchen.play_env.play_env import *

from robocasa.utils.mimicdroid_utils import (
    make_env,
    reset_to,
    get_zero_action,
    resample_actions,
    EnvArgs,
)

# Disable sites for cleaner rendering
macros.SHOW_SITES = False


def load_dataset_info(hdf5_path):
    """Load dataset information and return data structure."""
    with h5py.File(hdf5_path, "r") as f:
        if "data" in f:
            data = f["data"]
        else:
            data = f

        # Get episode names
        ep_names = sorted(list(data.keys()))
        print(colored(f"Found {len(ep_names)} episodes in {hdf5_path}", "green"))

        # Get first episode info for environment setup
        first_ep = ep_names[0]
        ep_group = data[first_ep]

        # Extract environment arguments
        env_args = (
            json.loads(data.attrs["env_args"])
            if "data" in f
            else json.loads(f.attrs["env_args"])
        )
        if isinstance(env_args, str):
            env_args = json.loads(env_args)

        # Extract episode metadata
        ep_meta = ep_group.attrs.get("ep_meta", {})
        if isinstance(ep_meta, str):
            ep_meta = json.loads(ep_meta)

        return hdf5_path, ep_names, env_args, ep_meta


def create_keys_info():
    """Create keys_info structure for environment setup."""
    # Default keys based on eval_casa.py
    image_keys = []
    proprio_keys = []
    action_keys = []
    low_dim_keys = []

    return {
        "image_keys": image_keys,
        "proprio_keys": proprio_keys,
        "action_keys": action_keys,
        "low_dim_keys": low_dim_keys,
    }


def replay_episode(env, hdf5_path, ep_name, env_args: EnvArgs, args: argparse.Namespace):
    """Replay a single episode from the dataset."""
    print(colored(f"Replaying episode: {ep_name}", "blue"))

    ep_group = h5py.File(hdf5_path, "r")["data"][ep_name]

    # Load states and actions
    states = ep_group["states"][:]  # shape: (T, state_dim)
    actions = ep_group["actions"][:]  # shape: (T, action_dim)

    # Optionally slow down (or speed up) playback by resampling the absolute targets.
    speed = getattr(args, "speed", 1.0)
    if speed != 1.0:
        orig_len = len(actions)
        actions = resample_actions(actions, speed)
        print(
            colored(
                f"Resampled actions for {speed}x speed: {orig_len} -> {len(actions)} steps",
                "magenta",
            )
        )

    # Get episode metadata
    ep_meta = ep_group.attrs.get("ep_meta", {})
    if isinstance(ep_meta, str):
        ep_meta = json.loads(ep_meta)

    # Get model file
    model_file = ep_group.attrs.get("model_file", None)

    # Get additional state info if available
    non_robot_qpos_idx = ep_group.attrs.get("non_robot_qpos_idx", None)
    qpos_state = ep_group.attrs.get("initial_qpos", None)

    # Create initial state
    initial_state = {
        "states": states[0],
        "ep_meta": ep_meta,
    }

    if model_file is not None:
        initial_state["model"] = model_file

    if non_robot_qpos_idx is not None:
        initial_state["non_robot_qpos_idx"] = non_robot_qpos_idx

    if qpos_state is not None:
        initial_state["qpos_state"] = qpos_state

    # Reset environment to initial state
    print(colored("Resetting environment to initial state...", "yellow"))
    reset_to(
        env,
        initial_state,
        replace_robot_joints=True,
        change_to_gr1="GR1" in env_args.robots,
    )

    video_frames = []

    # Optional warm-start: hold the first target so the arms servo to it before
    # the real trajectory begins. This removes the large initial tracking gap
    # caused by the robot resetting to its home pose rather than the demo's arm
    # pose (see analyze_ik_error.py for the diagnosis).
    warm_start_steps = getattr(args, "warm_start_steps", 0)
    if warm_start_steps > 0 and len(actions) > 0:
        print(
            colored(
                f"Warm-start: settling to first target for {warm_start_steps} steps...",
                "yellow",
            )
        )
        for _ in range(warm_start_steps):
            if env_args.render:
                env.render()
                time.sleep(0.05)
            obs, _, _, _ = env.step(actions[0])
            if args.save_videos:
                keys = [key for key in obs.keys() if "_image" in key]
                image = np.concatenate([obs[key][::-1][..., ::-1] for key in keys], axis=1)
                video_frames.append(image)

    # Replay actions
    print(colored(f"Replaying {len(actions)} actions...", "yellow"))

    for i, action in enumerate(actions):
        if env_args.render:
            # Render the environment
            env.render()
            time.sleep(0.05)  # Small delay for visualization

        # Step the environment
        obs, reward, done, info = env.step(action)
        if args.save_videos:
            keys = [key for key in obs.keys() if '_image' in key]
            image = np.concatenate([obs[key][::-1][..., ::-1] for key in keys], axis=1)
            video_frames.append(image)

        if hasattr(env_args, "verbose") and env_args.verbose:
            print(f"Step {i+1}/{len(actions)}: Action={action[:3]}... (first 3 dims)")

        # Check for early termination
        if done:
            print(colored(f"Episode terminated early at step {i+1}", "red"))
            break

    # Check success
    success = env._check_success()
    print(
        colored(f"Episode completed. Success: {success}", "green" if success else "red")
    )
    if args.save_videos:
        # Save to the requested video directory, named by episode + robot + speed
        # (+ warm-start) so different embodiments/settings never overwrite each other.
        video_dir = getattr(args, "video_dir", None) or "videos"
        os.makedirs(video_dir, exist_ok=True)
        robots = env_args.robots
        robot_tag = "_" + ("".join(robots) if isinstance(robots, list) else robots)
        speed_tag = f"_{speed}x" if speed != 1.0 else ""
        warm_tag = f"_warm{warm_start_steps}" if warm_start_steps > 0 else ""
        out_file = os.path.join(video_dir, f"{ep_name}{robot_tag}{speed_tag}{warm_tag}.mp4")
        video_writer = cv2.VideoWriter(out_file, cv2.VideoWriter_fourcc(*'mp4v'), 30, (video_frames[0].shape[1], video_frames[0].shape[0]))
        for frame in video_frames:
            video_writer.write(frame)
        video_writer.release()
        print(colored(f"Video saved to {out_file}", "green"))
    return success


def main():
    parser = argparse.ArgumentParser(
        description="Replay dataset trajectories from HDF5 files"
    )
    parser.add_argument(
        "--hdf5_path", type=str, required=True, help="Path to HDF5 dataset file"
    )
    parser.add_argument(
        "--episode_idx",
        type=int,
        default=0,
        help="Episode index to replay (default: 0)",
    )
    parser.add_argument(
        "--robots",
        type=str,
        default="DemoTwoHand",
        help="Robot type (default: DemoTwoHand)",
        choices=["DemoTwoHand", "GR1TwoHand"],
    )
    parser.add_argument(
        "--render", action="store_true", help="Render the environment during replay"
    )
    parser.add_argument("--verbose", action="store_true", help="Print verbose output")
    parser.add_argument(
        "--max_episodes",
        type=int,
        default=1,
        help="Maximum number of episodes to replay",
    )
    parser.add_argument(
        "--use_camera_obs",
        action="store_true",
        default=False,
        help="Whether to return camera observations",
    )
    parser.add_argument(
        "--save_videos",
        action="store_true",
        help="Whether to save a video of the episode",
    )
    parser.add_argument(
        "--video_dir",
        type=str,
        default="/mnt/ddn/tmp/replay_videos",
        help="Directory to save replay videos (named <episode>_<speed>x.mp4)",
    )
    parser.add_argument(
        "--warm_start_steps",
        type=int,
        default=0,
        help=(
            "Number of steps to hold the first action before replay, letting the "
            "arms servo to the first target from the home pose (removes the large "
            "initial IK tracking gap). 0 disables. Try 30."
        ),
    )
    parser.add_argument(
        "--reset_mode",
        type=str,
        default=None,
        help="Reset mode"
    )
    parser.add_argument(
        "--speed",
        type=float,
        default=1.0,
        help=(
            "Action replay speed multiplier. <1.0 slows down playback by inserting "
            "interpolated absolute targets (e.g. 0.7 for 0.7x speed), which improves "
            "IK tracking and reduces replay failures. 1.0 = original speed."
        ),
    )
    args = parser.parse_args()
    if args.save_videos:
        assert args.use_camera_obs, "Camera observations are required to save a video"

    # Validate inputs
    if not os.path.exists(args.hdf5_path):
        print(colored(f"Error: HDF5 file not found: {args.hdf5_path}", "red"))
        return

    args = parser.parse_args()

    # Validate inputs
    if not os.path.exists(args.hdf5_path):
        print(colored(f"Error: HDF5 file not found: {args.hdf5_path}", "red"))
        return

    # Load dataset information
    print(colored("Loading dataset information...", "yellow"))
    hdf5_path, ep_names, env_args, ep_meta = load_dataset_info(args.hdf5_path)

    # Create keys_info
    keys_info = create_keys_info()

    env_args_obj = EnvArgs(
        robots=args.robots,
        render=args.render,
        control_freq=20,
        controller="WHOLE_BODY_MINK_IK" if "GR1" in args.robots else None,
        use_camera_obs=args.use_camera_obs,
    )

    # Create environment
    print(colored("Creating environment...", "yellow"))
    env, env_kwargs = make_env(file_name=hdf5_path, env_args=env_args_obj)

    print(
        colored(
            f"Environment created successfully. Robot: {env_args_obj.robots}", "green"
        )
    )
    print(
        colored(
            f"Controller: {env_args_obj.controller if env_args_obj.controller else 'Default'}",
            "green",
        )
    )

    # Replay episodes
    success_count = 0
    total_episodes = min(args.max_episodes, len(ep_names))

    for i in range(total_episodes):
        ep_idx = (args.episode_idx + i) % len(ep_names)
        ep_name = ep_names[ep_idx]

        print(colored(f"\n{'='*50}", "cyan"))
        print(colored(f"Episode {i+1}/{total_episodes}: {ep_name}", "cyan"))
        print(colored(f"{'='*50}", "cyan"))

        replay_episode(env, hdf5_path, ep_name, env_args_obj, args)

    # Print summary
    print(colored(f"\n{'='*50}", "cyan"))


if __name__ == "__main__":
    main()
