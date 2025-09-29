# 文件路径: isaaclab_tasks/manager_based/manipulation/reach/mdp/cus_command.py

from __future__ import annotations

import torch
from dataclasses import MISSING
from typing import TYPE_CHECKING

from isaaclab.utils import math as math_utils
from isaaclab.managers import CommandTerm, CommandTermCfg
from isaaclab.utils import configclass

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv




class ConstantPoseCommand(CommandTerm):
    """
    在整个回合中提供一个恒定的目标姿态指令。
    机器人将从其初始状态移动到这个固定的目标。
    """
    cfg: "ConstantPoseCommandCfg"

    def __init__(self, cfg: "ConstantPoseCommandCfg", env: ManagerBasedRLEnv):
        """初始化指令项。"""
        super().__init__(cfg, env)
        # 用于存储指令的缓冲区。形状: (num_envs, 7) -> [pos_x, pos_y, pos_z, quat_x, quat_y, quat_z, quat_w]
        self.buffer = torch.zeros(self.num_envs, 7, device=self.device)
        # 指令在重置时通过 _resample_command 设置一次即可。

    def _resample_command(self, env_ids: torch.Tensor):
        """为指定的环境将指令缓冲区设置为固定的目标姿态。"""
        # -- 设置目标位置
        target_pos = torch.tensor([
            self.cfg.target_pose.pos_x,
            self.cfg.target_pose.pos_y,
            self.cfg.target_pose.pos_z
        ], device=self.device)
        self.buffer[env_ids, :3] = target_pos.expand(len(env_ids), 3)

        # -- 设置目标姿态 (欧拉角 -> 四元数)
        target_roll = torch.full((len(env_ids),), self.cfg.target_pose.roll, device=self.device)
        target_pitch = torch.full((len(env_ids),), self.cfg.target_pose.pitch, device=self.device)
        target_yaw = torch.full((len(env_ids),), self.cfg.target_pose.yaw, device=self.device)
        q_target_wxyz = math_utils.quat_from_euler_xyz(target_roll, target_pitch, target_yaw)
        # 转换为 xyzw 格式存入缓冲区
        self.buffer[env_ids, 3:7] = q_target_wxyz[:, [1, 2, 3, 0]]

    def _update_command(self):
        """指令是恒定的，因此每个步骤不需要更新。"""
        # 缓冲区已在重置时由 _resample_command 设置好。
        pass

    def _update_metrics(self):
        """没有需要更新的指标。"""
        pass

    @property
    def command(self) -> torch.Tensor:
        """返回当前指令。"""
        return self.buffer

@configclass
class ConstantPoseCommandCfg(CommandTermCfg):
    """
    `ConstantPoseCommand` 的配置类。
    """
    class_type: type = ConstantPoseCommand
    # ▼▼▼ 添加这个缺失的字段 ▼▼▼
    resampling_time_range: tuple[float, float] = (0.0, 0.0) # 不需要使用，但为了通过验证必须存在

    asset_name: str = MISSING
    debug_vis: bool = False

    @configclass
    class Pose:
        """用于定义一个固定姿态的配置。"""
        pos_x: float = 0.0
        pos_y: float = 0.0
        pos_z: float = 0.0
        roll: float = 0.0
        pitch: float = 0.0
        yaw: float = 0.0

    target_pose: Pose = Pose()

    # -- 以上新增的指令类 --