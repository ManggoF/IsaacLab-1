# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
import isaaclab.utils.math as math_utils
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.sensors import FrameTransformer
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils.math import subtract_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv, ManagerBasedEnv
    


def object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The position of the object in the robot's root frame."""
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    object_pos_w = object.data.root_pos_w[:, :3]
    object_pos_b, _ = subtract_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, object_pos_w)
    return object_pos_b


def root_lin_vel(
    env: ManagerBasedEnv, frame: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Asset root linear velocity in a specified frame.

    Args:
        env: The environment.
        frame: The frame in which to express the velocity. Can be "world", "root", or "{asset_name}_root".
        asset_cfg: The SceneEntity associated with this observation.

    Returns:
        The asset's root linear velocity in the specified frame.
    """
    # extract the used quantities
    asset: RigidObject = env.scene[asset_cfg.name]
    lin_vel_w = asset.data.root_lin_vel_w

    if frame == "world":
        return lin_vel_w
    elif frame == "root":
        return asset.data.root_lin_vel_b
    else:
        # assume the frame is the name of another asset's root
        if frame.endswith("_root"):
            other_asset_name = frame[: -len("_root")]
        else:
            other_asset_name = frame
        # get the other asset's data
        other_asset: RigidObject = env.scene[other_asset_name]
        # transform the velocity from world to the other asset's root frame
        return math_utils.quat_apply_inverse(other_asset.data.root_quat_w, lin_vel_w)
    
    
def target_object_position_in_robot_root_frame(
    env: ManagerBasedRLEnv,
    target_offset: tuple[float, float, float],
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """The desired tool-center-point position in the robot's root frame.
    
    The target is defined as the object's position with a vertical offset.
    """
    robot: Articulation = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    
    # 1. 获取物体在世界坐标系中的位置
    object_pos_w = object.data.root_pos_w
    
    # 2. 计算出我们期望的目标点 (在物体上方)
    target_pos_w = object_pos_w + torch.tensor(target_offset, device=env.device)
    
    # 3. 将这个世界坐标系下的目标点，转换到机器人基座坐标系下
    target_pos_b, _ = subtract_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, target_pos_w
    )
    
    return target_pos_b    