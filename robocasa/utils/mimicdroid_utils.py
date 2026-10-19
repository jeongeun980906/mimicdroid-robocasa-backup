# Description: Utility functions for loading and processing data from the MimicDroid dataset
# This module contains functions moved from casa_utils.py to make replay_dataset.py isolated

import os
import json
import h5py
import numpy as np
import xml.etree.ElementTree as ET
from pathlib import Path
from lxml import etree
from dataclasses import dataclass
from typing import Optional, Union, List

from scipy.spatial.transform import Rotation, Slerp

import robosuite
from robosuite.utils.binding_utils import MjSimState
from robosuite.controllers import load_composite_controller_config

import robocasa
from robocasa import PLAY_TASK_NAME_TO_CLASS
import robocasa.macros as macros

def fix_asset_paths_relative_to_robocasa(
    xml_string, robocasa_module, robocasa_marker="robocasa/models/assets"
):
    """
    Replace all asset file paths in the xml string so that the prefix up to 'robocasa'
    is replaced with the robocasa installation path.

    Args:
        xml_string (str): Original MJCF XML as string
        robocasa_module: The imported `robocasa` module (or any module within it)
        robocasa_marker (str): Folder name to identify the point in path replacement
    """
    # Parse and fix XML
    root = etree.fromstring(xml_string)
    for tag in ["mesh", "texture", "heightfield"]:
        for elem in root.findall(f".//{tag}"):
            file_attr = elem.get("file")
            if file_attr:
                # Replace only if absolute and contains old robocasa path
                if file_attr.startswith("/"):
                    # parts = Path(file_attr).parts
                    # find out what is the index of the robocasa marker
                    idx = file_attr.find(robocasa_marker)
                    if idx != -1:
                        curr_home_path = "/".join(robocasa.__file__.split("/")[:-2])
                        rel_path = file_attr[idx:]
                        new_path = os.path.join(curr_home_path, rel_path)
                        new_path = Path(new_path).resolve()
                        elem.set("file", str(new_path))

    return etree.tostring(root, pretty_print=True).decode("utf-8")


@dataclass
class EnvArgs:
    """Environment arguments for creating robosuite environments."""

    robots: Union[str, List[str]] = "DemoTwoHand"
    controller: Optional[str] = None
    task_name: Optional[str] = None
    reset_mode: Optional[str] = None
    render: bool = False
    control_freq: int = 20
    use_camera_obs: bool = False
    camera_segmentations: Optional[str] = "instance"


def get_env_args_from_dataset(dataset_path):
    """Extract environment arguments from dataset."""
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    env_args = (
        json.loads(f["data"].attrs["env_args"])
        if "data" in f
        else json.loads(f.attrs["env_args"])
    )
    if isinstance(env_args, str):
        env_args = json.loads(env_args)  # double leads to dict type
    f.close()
    return env_args


def get_env_meta_from_dataset(dataset_path, index=0):
    """Extract environment metadata from dataset."""
    dataset_path = os.path.expanduser(dataset_path)
    f = h5py.File(dataset_path, "r")
    data = f["data"] if "data" in f else f
    keys = list(data.keys())
    env_meta = data[keys[index]].attrs["ep_meta"]
    env_meta = json.loads(env_meta)
    assert isinstance(env_meta, dict), f"Expected dict type but got {type(env_meta)}"
    return env_meta


def load_controller_config(controller, robot, control_type, ref_frame):
    """Load and configure controller settings."""
    controller_config = load_composite_controller_config(
        controller=controller,
        robot=robot,
    )
    if "right" in controller_config["body_parts"]:
        controller_config["body_parts"]["right"]["input_type"] = control_type
        controller_config["body_parts"]["right"]["input_ref_frame"] = ref_frame
    if "left" in controller_config["body_parts"]:
        controller_config["body_parts"]["left"]["input_type"] = control_type
        controller_config["body_parts"]["left"]["input_ref_frame"] = ref_frame
    if "WHOLE_BODY" in controller_config["type"]:
        controller_config["composite_controller_specific_configs"][
            "ik_input_ref_frame"
        ] = ref_frame
        controller_config["composite_controller_specific_configs"][
            "ik_input_type"
        ] = control_type
    return controller_config


def replace_robot_tag(new_model_xml, old_model_xml):
    """
    Replace robot tags in the XML.

    Args:
        new_model_xml: str (XML content of the new model)
        old_model_xml: str (XML content of the old model)

    Returns:
        str: Modified XML content as a string.
    """
    # Parse XML content
    new_root = ET.fromstring(new_model_xml)
    old_root = ET.fromstring(old_model_xml)

    # Function to filter and transfer robot0 elements
    def transfer_robot0_elements(new_parent, old_parent):
        # Remove existing robot0 elements in new_parent
        if new_parent is not None:
            for elem in list(new_parent):
                if "robot0" in elem.attrib.get("name", ""):
                    new_parent.remove(elem)

        if old_parent is not None:
            if new_parent is None:
                # add a new parent element if it doesn't exist to the
                # Add robot0 elements from old_parent to new_parent as a child of <mujoco> tag
                new_parent = ET.SubElement(new_root, old_parent.tag)
            for elem in old_parent:
                if "robot0" in elem.attrib.get("name", ""):
                    new_parent.append(elem)

    # Update <actuator> elements
    new_actuator = new_root.find("actuator")
    old_actuator = old_root.find("actuator")
    transfer_robot0_elements(new_actuator, old_actuator)

    # Update <asset> elements
    new_asset = new_root.find("asset")
    old_asset = old_root.find("asset")
    transfer_robot0_elements(new_asset, old_asset)

    new_contact = new_root.find("contact")
    old_contact = old_root.find("contact")
    transfer_robot0_elements(new_contact, old_contact)

    new_robot0_base = None
    for body in new_root.findall(".//body"):
        if body.attrib.get("name") == "robot0_floating_base":
            new_robot0_base = body
            break

    # Locate "robot0_floating_base" in old model
    old_robot0_floating_base = None
    for body in old_root.findall(".//body"):
        if body.attrib.get("name") == "robot0_base":
            old_robot0_floating_base = body
            break

    assert new_robot0_base is not None, "robot0_floating_base must exist in new model"
    assert old_robot0_floating_base is not None, "robot0_base must exist in old model"
    # Replace the contents of "robot0_base" with "robot0_floating_base" if both exist
    if new_robot0_base is not None and old_robot0_floating_base is not None:
        # Clear the current children of new_robot0_base
        new_robot0_base.clear()

        # Copy all attributes from old_robot0_floating_base to new_robot0_base
        new_robot0_base.attrib = old_robot0_floating_base.attrib

        # Copy all child elements
        for elem in old_robot0_floating_base:
            new_robot0_base.append(elem)

    # copy the geom name from old model to new model named robot0_floor in the worldbody tag
    # we want to put it in the same location in the new model
    new_worldbody = new_root.find("worldbody")
    old_worldbody = old_root.find("worldbody")
    for geom in old_worldbody.findall(".//geom"):
        if geom.attrib.get("name") == "robot0_floor":
            index = list(old_worldbody).index(geom)
            new_worldbody.insert(index, geom)

    # Convert back to string
    return ET.tostring(new_root, encoding="unicode")


def resample_actions(actions, speed, pose_blocks=(0, 6), gripper_idx=(12, 13)):
    """Resample an absolute-action trajectory to play back at a different speed.

    The MimicDroid datasets use absolute pose targets (input_type == "absolute"),
    so slowing playback down is done by inserting interpolated targets between the
    recorded ones. At ``speed=0.7`` the robot is given ~1/0.7x as many targets, each
    advancing only 0.7 of the way, so it moves more slowly and the IK controller
    tracks each target more accurately (fewer replay failures).

    Args:
        actions (np.ndarray): (T, action_dim) recorded absolute actions.
        speed (float): playback speed multiplier. <1.0 slows down, >1.0 speeds up.
        pose_blocks (tuple[int]): start index of each 6-DoF eef pose block
            (3 position dims + 3 axis-angle rotation dims). Default matches the
            14-dim two-hand layout: right pose [0:6], left pose [6:12].
        gripper_idx (tuple[int]): indices of discrete gripper command dims, held
            with nearest-neighbor (never blended) to preserve open/close commands.

    Returns:
        np.ndarray: (new_T, action_dim) resampled absolute actions.
    """
    actions = np.asarray(actions, dtype=np.float64)
    T, action_dim = actions.shape
    if speed == 1.0 or T < 2:
        return actions

    # Validate that the assumed layout fits this action dimension. Fall back to a
    # pure linear interpolation (no slerp / gripper handling) for unexpected dims.
    layout_ok = action_dim == 14 and all(b + 6 <= action_dim for b in pose_blocks)

    # New timeline: fractional indices into the original [0, T-1] trajectory.
    # Use linspace so the first and last recorded targets are hit exactly (the
    # trajectory still reaches its final goal), while the average step spacing
    # stays ~= speed.
    new_T = int(round((T - 1) / speed)) + 1
    src_times = np.linspace(0.0, T - 1, new_T)

    new_actions = np.empty((new_T, action_dim), dtype=np.float64)
    lo = np.floor(src_times).astype(int)
    hi = np.minimum(lo + 1, T - 1)
    frac = (src_times - lo)[:, None]

    # Default: linear interpolation for every dim.
    new_actions[:] = (1.0 - frac) * actions[lo] + frac * actions[hi]

    if layout_ok:
        # Proper slerp for the 3-dim axis-angle rotation part of each pose block.
        for b in pose_blocks:
            rot_lo = Rotation.from_rotvec(actions[lo, b + 3 : b + 6])
            rot_hi = Rotation.from_rotvec(actions[hi, b + 3 : b + 6])
            for j in range(new_T):
                if lo[j] == hi[j]:
                    new_actions[j, b + 3 : b + 6] = actions[lo[j], b + 3 : b + 6]
                    continue
                slerp = Slerp([0.0, 1.0], Rotation.concatenate([rot_lo[j], rot_hi[j]]))
                new_actions[j, b + 3 : b + 6] = slerp(frac[j, 0]).as_rotvec()

        # Grippers: nearest-neighbor so open/close commands are never blended.
        nearest = np.clip(np.round(src_times).astype(int), 0, T - 1)
        for g in gripper_idx:
            new_actions[:, g] = actions[nearest, g]

    return new_actions


def get_zero_action(env):
    """Get zero action for the environment."""
    active_robot = env.robots[0]
    if env.action_dim == 14:
        arms = ["left", "right"]
        zero_action_dict = {}
        for arm in arms:
            # controller has absolute actions, so we need to set the initial action to be the current position
            zero_action = np.zeros(7)
            if active_robot.part_controllers[arm].input_type == "absolute":
                zero_action = robosuite.utils.control_utils.convert_delta_to_abs_action(
                    zero_action, active_robot, arm, env
                )
            zero_action_dict[f"{arm}"] = zero_action[: zero_action.shape[0] - 1]
            zero_action_dict[f"{arm}_gripper"] = zero_action[zero_action.shape[0] - 1 :]
        zero_action_dict["base_mode"] = -1
        zero_action_dict["base"] = np.zeros(3)
        zero_action = active_robot.create_action_vector(zero_action_dict)
    else:
        zero_action = np.zeros(env.action_dim)
        if active_robot.part_controllers[arm].input_type == "absolute":
            zero_action = robosuite.utils.control_utils.convert_delta_to_abs_action(
                zero_action, active_robot, arm, env
            )
    return zero_action


def make_env(file_name, env_args: EnvArgs):
    """Create environment from dataset file."""
    dataset_env_args = get_env_args_from_dataset(dataset_path=file_name)
    env_meta = get_env_meta_from_dataset(dataset_path=file_name, index=0)
    dataset_controller_config = dataset_env_args["env_kwargs"]["controller_configs"]

    if env_args.controller is not None:
        controller_config = load_controller_config(
            controller=env_args.controller,
            robot=env_args.robots
            if isinstance(env_args.robots, str)
            else env_args.robots[0],
            control_type=dataset_controller_config["body_parts"]["right"]["input_type"],
            ref_frame=dataset_controller_config["body_parts"]["right"][
                "input_ref_frame"
            ],
        )
    else:
        if macros.VERBOSE:
            print("No controller specified. Using default controller from the dataset")
        controller_config = dataset_controller_config

    env_name = dataset_env_args["env_name"]
    if macros.VERBOSE:
        print(f"{env_name=}")
    if env_args.task_name is not None:
        env_name = PLAY_TASK_NAME_TO_CLASS[env_args.task_name]
    env_kwargs = dataset_env_args["env_kwargs"]

    env_kwargs["eval_mode"] = env_args.reset_mode
    env_kwargs["env_name"] = env_name
    env_kwargs[
        "ep_meta"
    ] = env_meta  # this should ideally reduce exploration for finding correct set of objects
    env_kwargs["robots"] = (
        env_args.robots if isinstance(env_args.robots, list) else [env_args.robots]
    )
    env_kwargs["controller_configs"] = controller_config
    env_kwargs.pop("has_renderer", None)
    env_kwargs.pop("use_camera_obs", None)
    env_kwargs.pop("renderer", None)
    env_kwargs.pop("camera_segmentations", None)

    env = robosuite.make(
        has_renderer=env_args.render,
        use_camera_obs=env_args.use_camera_obs,
        renderer="mjviewer",
        camera_segmentations=env_args.camera_segmentations,
        control_freq=env_args.control_freq,
        **env_kwargs,
    )
    return env, env_kwargs


def reset_to(env, state, replace_robot_joints=True, change_to_gr1=False):
    """
    Reset to a specific simulator state.

    Args:
        env: The environment to reset
        state (dict): current simulator state that contains one or more of:
            - states (np.ndarray): initial state of the mujoco environment
            - model (str): mujoco scene xml
        replace_robot_joints (bool): whether to replace robot joints
        change_to_gr1 (bool): whether to change to GR1 robot

    Returns:
        None
    """
    should_ret = False
    if "model" in state:
        if state.get("ep_meta", None) is not None:
            # set relevant episode information
            ep_meta = state["ep_meta"]
            if isinstance(ep_meta, str):
                ep_meta = json.loads(ep_meta)
        else:
            ep_meta = {}
        if hasattr(env, "set_attrs_from_ep_meta"):  # older versions had this function
            env.set_attrs_from_ep_meta(ep_meta)
        elif hasattr(env, "set_ep_meta"):  # newer versions
            env.set_ep_meta(ep_meta)
        else:
            raise ValueError(
                "Environment does not have set_ep_meta or set_attrs_from_ep_meta function"
            )
        # this reset is necessary.
        # while the call to env.reset_from_xml_string does call reset,
        # that is only a "soft" reset that doesn't actually reload the model.
        env.reset()
        robosuite_version_id = int(robosuite.__version__.split(".")[1])
        # we need to first update state["model"] to replace the robot tag with the current robot
        curr_xml = env.sim.model.get_xml()

        if robosuite_version_id <= 3:
            from robosuite.utils.mjcf_utils import postprocess_model_xml

            xml = postprocess_model_xml(state["model"])
        else:
            # v1.4 and above use the class-based edit_model_xml function
            xml = env.edit_model_xml(state["model"])

        if change_to_gr1:
            xml = replace_robot_tag(new_model_xml=xml, old_model_xml=curr_xml)

        xml = fix_asset_paths_relative_to_robocasa(xml_string=xml, robocasa_module=robocasa)
        env.reset_from_xml_string(xml)
        # env.sim.reset(): resets the robot back to some position which has collision with the table. Change the xml?
        env.sim.reset()
        env.robots[0].reset()
        env.sim.forward()

    if "states" in state:
        if replace_robot_joints:
            if macros.VERBOSE:
                print(f"Replacing robot joints")
            robot_indices = env.robots[0]._ref_joint_pos_indexes
            other_indices = set(
                range(env.sim.get_state().qpos.flatten().shape[0])
            ) - set(robot_indices)
            other_indices = sorted(list(other_indices))

            non_robot_qpos_idx = state["non_robot_qpos_idx"]
            assert len(non_robot_qpos_idx) == len(
                other_indices
            ), f"Mismatch in non_robot_qpos_idx: {len(non_robot_qpos_idx)} != {len(other_indices)}"
            qpos_state = state["qpos_state"]
            time = env.sim.data.time
            qvel = env.sim.get_state().qvel.flatten().copy()
            qpos = env.sim.get_state().qpos.flatten().copy()
            # copy over everything except the robot state
            for curr_idx, state_idx in zip(other_indices, non_robot_qpos_idx):
                qpos[curr_idx] = qpos_state[state_idx]
            curr_state = MjSimState(qpos=qpos, qvel=qvel, time=time)
            env.sim.set_state_from_flattened(curr_state.flatten())
            env.sim.forward()
            should_ret = True
        else:
            env.sim.set_state_from_flattened(state["states"])
        env.sim.forward()
        should_ret = True
        if replace_robot_joints:
            env.robots[0].reset()
            env.sim.forward()

    # update state as needed
    if hasattr(env, "update_sites"):
        # older versions of environment had update_sites function
        env.update_sites()
    if hasattr(env, "update_state"):
        # later versions renamed this to update_state
        env.update_state()

    return None
