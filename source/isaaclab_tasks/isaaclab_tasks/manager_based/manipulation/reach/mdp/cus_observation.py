# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def object_ee_position(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, ee_asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """
    计算并返回一个物体（例如球）相对于机器人末端执行器（例如勺子）的局部位置向量。
    这个观察对于需要精确操控物体的任务至关重要。
    """
    # 提取球和机器人对象
    obj: RigidObject = env.scene[asset_cfg.name]
    robot: Articulation = env.scene[ee_asset_cfg.name]

    # 获取球在世界坐标系中的位置
    obj_pos_w = obj.data.root_state_w[:, :3]
    # 获取末端执行器（Link6）在世界坐标系中的位置
    ee_pos_w = robot.data.body_pos_w[:, ee_asset_cfg.body_ids[0]]

    # 计算相对位置向量
    relative_pos = obj_pos_w - ee_pos_w

    return relative_pos
