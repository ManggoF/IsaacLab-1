# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import RigidObject, Articulation
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import FrameTransformer
from isaaclab.utils.math import combine_frame_transforms

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv



def object_is_lifted(
    env: ManagerBasedRLEnv, minimal_height: float, object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Reward the agent for lifting the object above the minimal height."""
    object: RigidObject = env.scene[object_cfg.name]
    return torch.where(object.data.root_pos_w[:, 2] > minimal_height, 1.0, 0.0)


def object_ee_distance(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """Reward the agent for reaching the object using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    object: RigidObject = env.scene[object_cfg.name]
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    # Target object position: (num_envs, 3)
    cube_pos_w = object.data.root_pos_w
    # End-effector position: (num_envs, 3)
    ee_w = ee_frame.data.target_pos_w[..., 0, :]
    # Distance of the end-effector to the object: (num_envs,)
    object_ee_distance = torch.norm(cube_pos_w - ee_w, dim=1)

    return 1 - torch.tanh(object_ee_distance / std)


def object_goal_distance(
    env: ManagerBasedRLEnv,
    std: float,
    minimal_height: float,
    command_name: str,
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """Reward the agent for tracking the goal pose using tanh-kernel."""
    # extract the used quantities (to enable type-hinting)
    robot: RigidObject = env.scene[robot_cfg.name]
    object: RigidObject = env.scene[object_cfg.name]
    command = env.command_manager.get_command(command_name)
    # compute the desired position in the world frame
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b)
    # distance of the end-effector to the object: (num_envs,)
    distance = torch.norm(des_pos_w - object.data.root_pos_w, dim=1)
    # rewarded if the object is lifted above the threshold
    return (object.data.root_pos_w[:, 2] > minimal_height) * (1 - torch.tanh(distance / std))



def reaching_goal_bounded(
    env: ManagerBasedRLEnv,
    std: float, # 我们用 std 来控制奖励的“敏感范围”
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
) -> torch.Tensor:
    """
    A smooth AND BOUNDED reward for reaching the object.
    This reward is bounded between 0 and 1 using a Gaussian kernel.
    """
    # 1. 目标点就是在物体质心
    target_pos_w = env.scene[object_cfg.name].data.root_pos_w
    
    # 2. 获取当前TCP位置
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    tcp_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    
    # 3. 计算距离
    distance = torch.norm(target_pos_w - tcp_pos_w, dim=1)
    
    # 4. 使用高斯核函数，将距离转化为一个 [0, 1] 之间的平滑奖励
    #    距离越近，奖励越接近1；距离越远，奖励越接近0。
    #    它永远不会是负数，也永远不会超过1。
    reward = torch.exp(-torch.square(distance / std))
    
    return reward
    



def approach_object_vertically(
    env: ManagerBasedRLEnv,
    std: float,
    horizontal_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """当水平对齐时，奖励机械臂在垂直方向上接近物块。"""
    # 物块位置
    object_pos_w = env.scene[object_cfg.name].data.root_pos_w
    # 末端执行器位置
    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]

    # 计算水平和垂直距离
    horizontal_dist = torch.norm(object_pos_w[:, :2] - ee_pos_w[:, :2], dim=1)
    vertical_dist = torch.abs(object_pos_w[:, 2] - ee_pos_w[:, 2])

    # 只有当水平距离小于阈值时，才给予垂直接近的奖励
    # 这样可以防止机械臂在没有对齐时就贸然下降
    reward = torch.where(
        horizontal_dist < horizontal_threshold,
        1 - torch.tanh(vertical_dist / std),
        torch.zeros_like(vertical_dist)
    )
    return reward

def grasp_success(
    env: ManagerBasedRLEnv,
    gripper_close_bonus: float,
    object_ee_distance_threshold: float,
    gripper_joint_closed_threshold: float,
    horizontal_alignment_threshold: float, # << 新增参数
    vertical_alignment_threshold: float,   # << 新日志参数
    robot_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    一个有条件的、分阶段的抓取奖励函数。
    它只在机械臂处于正确的预抓取姿态时，才奖励闭合夹爪的行为。
    """
    # --- 1. 检查是否满足预抓取条件 ---
    object_pos = env.scene[object_cfg.name].data.root_pos_w
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    
    # 检查水平对齐
    horizontal_dist = torch.norm(object_pos[:, :2] - ee_pos[:, :2], dim=1)
    is_aligned_horizontally = horizontal_dist < horizontal_alignment_threshold
    
    # 检查垂直对齐 (末端需要在物体上方一个很近的距离)
    vertical_dist = torch.abs(object_pos[:, 2] - ee_pos[:, 2])
    is_aligned_vertically = vertical_dist < vertical_alignment_threshold
    
    # 只有同时满足水平和垂直对齐，才认为处于“可抓取”状态
    is_in_pre_grasp_pose = is_aligned_horizontally & is_aligned_vertically

    # --- 2. 如果满足条件，则计算抓取奖励 ---
    robot: Articulation = env.scene[robot_cfg.name]
    gripper_joint_idx = robot.joint_names.index("finger_joint")
    gripper_pos = robot.data.joint_pos[:, gripper_joint_idx]

    # 奖励闭合夹爪的“意图”：只要处于预抓取姿态，闭合夹爪就能获得平滑的奖励
    # 这样可以鼓励它在正确的位置尝试闭合
    closing_intent_reward = gripper_pos / gripper_joint_closed_threshold
    closing_intent_reward = torch.clamp(closing_intent_reward, 0.0, 1.0)
    
    # 检查物理上是否抓取成功
    is_gripper_physically_closed = gripper_pos > gripper_joint_closed_threshold
    is_object_in_gripper = horizontal_dist < object_ee_distance_threshold # 用水平距离判断更鲁棒
    
    # 给予成功的“巨额”奖励
    success_bonus = torch.where(is_gripper_physically_closed & is_object_in_gripper, gripper_close_bonus, 0.0)

    # --- 3. 最终奖励 ---
    # 只有当处于预抓取姿态时，才激活所有抓取相关的奖励
    # 这从根本上杜绝了在错误位置闭合夹爪的行为
    final_reward = torch.where(
        is_in_pre_grasp_pose,
        closing_intent_reward + success_bonus, # 既奖励尝试，也奖励成功
        torch.zeros_like(success_bonus)
    )
    
    return final_reward



def horizontal_alignment(
    env: ManagerBasedRLEnv,
    std: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """奖励机械臂末端在水平面上(X-Y平面)与物块对齐。"""
    # 物块位置 (num_envs, 3)
    object_pos_w = env.scene[object_cfg.name].data.root_pos_w
    # 末端执行器位置 (num_envs, 3)
    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]

    # 只计算 X-Y 平面上的距离
    horizontal_dist = torch.norm(object_pos_w[:, :2] - ee_pos_w[:, :2], dim=1)

    # 使用高斯核函数计算奖励，距离越近奖励越高，这是一个比tanh更平滑的塑形
    reward = torch.exp(-torch.square(horizontal_dist / std))
    return reward


def approach_from_top(
    env: ManagerBasedRLEnv,
    std: float,
    horizontal_threshold: float,
    object_cfg: SceneEntityCfg = SceneEntityCfg("object"),
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """当水平对齐时，奖励机械臂在垂直方向上从上方接近物块。"""
    # 物块位置
    object_pos_w = env.scene[object_cfg.name].data.root_pos_w
    # 末端执行器位置
    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]

    # 计算水平和垂直距离
    horizontal_dist = torch.norm(object_pos_w[:, :2] - ee_pos_w[:, :2], dim=1)
    # Z轴距离，我们只关心末端在物体上方的情况
    vertical_dist = ee_pos_w[:, 2] - object_pos_w[:, 2]

    # 关键逻辑：
    # 1. 只有当水平距离小于阈值时 (已经对齐了)
    # 2. 并且末端在物体上方 (vertical_dist > 0)
    # 才给予垂直接近的奖励
    is_aligned_horizontally = horizontal_dist < horizontal_threshold
    is_above_object = vertical_dist > 0
    
    # 只有同时满足两个条件，才计算奖励
    eligible_envs = is_aligned_horizontally & is_above_object
    
    # 使用高斯核函数计算垂直奖励
    vertical_reward = torch.exp(-torch.square(vertical_dist / std))
    
    # 将不满足条件的智能体的奖励置为0
    reward = torch.where(eligible_envs, vertical_reward, 0.0)
    return reward


# def penalize_object_velocity(
#     env: ManagerBasedRLEnv,
#     object_cfg: SceneEntityCfg = SceneEntityCfg("object")
# ) -> torch.Tensor:
#     """惩罚物块的高速度，防止机器人将其撞飞。"""
#     object_vel = env.scene[object_cfg.name].data.root_lin_vel_w
#     # 计算速度的模长
#     object_speed = torch.norm(object_vel, dim=1)
#     # 速度越高，惩罚越大
#     return torch.square(object_speed)

def penalize_lingering(
    env: ManagerBasedRLEnv,
    horizontal_threshold: float,
    vertical_threshold: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
    object_cfg: SceneEntityCfg = SceneEntityCfg("object")
) -> torch.Tensor:
    """Penalizes the robot for hovering in the pre-grasp pose for too long."""
    object_pos_w = env.scene[object_cfg.name].data.root_pos_w
    ee_pos_w = env.scene[ee_frame_cfg.name].data.target_pos_w[..., 0, :]
    
    # 检查是否处于预抓取姿态
    horizontal_dist = torch.norm(object_pos_w[:, :2] - ee_pos_w[:, :2], dim=1)
    vertical_dist = torch.abs(ee_pos_w[:, 2] - object_pos_w[:, 2])
    
    is_lingering = (horizontal_dist < horizontal_threshold) & (vertical_dist < vertical_threshold)
    
    # 检查夹爪是否是张开的（即，没有尝试抓取）
    robot: Articulation = env.scene["robot"]
    gripper_joint_idx = robot.joint_names.index("finger_joint")
    is_gripper_open = robot.data.joint_pos[:, gripper_joint_idx] < 0.1 # 假设0.1以下为张开

    # 只有当处于预备姿态且夹爪张开时，才施加惩罚
    penalty = torch.where(is_lingering & is_gripper_open, -1.0, 0.0)
    return penalty

def penalize_ee_below_height(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    ee_frame_cfg: SceneEntityCfg = SceneEntityCfg("ee_frame"),
) -> torch.Tensor:
    """
    Penalizes the end-effector for being below a certain height.
    The lower it goes, the larger the penalty.
    """
    ee_frame: FrameTransformer = env.scene[ee_frame_cfg.name]
    
    # 获取TCP在世界坐标系中的位置
    tcp_pos_w = ee_frame.data.target_pos_w[..., 0, :]
    
    # 获取TCP的Z轴坐标 (高度)
    tcp_height = tcp_pos_w[:, 2]
    
    # 计算TCP低于最小高度的量 (如果高于，则为负数或零)
    height_difference = tcp_height - minimum_height
    
    # 核心逻辑：
    # 1. 使用 torch.min(height_difference, 0.0) 来只保留负值部分，即低于安全线的部分。
    # 2. 乘以 -1，将这个负的差值（例如-0.03米）变成一个正的惩罚值（0.03）。
    #    这样，低于安全线越深，惩罚值越大。
    penalty = -torch.min(height_difference, torch.tensor(0.0, device=env.device))
    
    # 返回这个惩罚值
    return penalty
