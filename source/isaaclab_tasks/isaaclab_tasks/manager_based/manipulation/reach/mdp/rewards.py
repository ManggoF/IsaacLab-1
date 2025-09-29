# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

import torch
from typing import TYPE_CHECKING

from isaaclab.assets import Articulation
from isaaclab.assets import RigidObject, AssetBase
from isaaclab.managers import SceneEntityCfg, ManagerTermBase
from isaaclab.utils.math import combine_frame_transforms, quat_error_magnitude, quat_mul, quat_apply, euler_xyz_from_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv
    from isaaclab.managers import RewardTermCfg

def spoon_position_command_error(
    env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg, offset: tuple[float, float, float]
) -> torch.Tensor:
    """
    【修正版】
    惩罚“勺子中心”与目标位置的跟踪误差 (L2-norm)。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 计算目标在世界坐标系中的位置 (这部分不变)
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)

    # 【核心修正】计算勺子中心在世界坐标系中的位置
    link_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]
    link_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    offset_tensor = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(env.num_envs, 1)
    curr_spoon_pos_w = link_pos_w + quat_apply(link_quat_w, offset_tensor)

    # 返回勺子中心与目标的距离
    return torch.norm(curr_spoon_pos_w - des_pos_w, dim=1)

def spoon_position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg, offset: tuple[float, float, float]
) -> torch.Tensor:
    """
    【修正版】
    奖励“勺子中心”接近目标位置 (tanh核函数)。
    """
    # 我们可以直接复用上面的函数来计算距离，避免代码重复
    distance = spoon_position_command_error(env, command_name, asset_cfg, offset)
    return 1 - torch.tanh(distance / std)

def ball_position_command_error(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg, ball_cfg: SceneEntityCfg, offset: tuple[float, float, float]
) -> torch.Tensor:
    """
    惩罚“小球”与“勺子中心”之间的距离 (L2-norm)。
    这可以作为一个塑形奖励，鼓励机器人用勺子去接近球。

    Args:
        env: 环境实例。
        asset_cfg: 机器人资产的配置，需要指定末端连杆的 body_ids。
        ball_cfg: 场景中的小球实体。
        offset: 从末端连杆原点到勺子中心的三维偏移量。

    Returns:
        一个张量，表示每个环境中“小球”与“勺子中心”之间的欧几里得距离。
    """
    # 1. 计算勺子中心在世界坐标系中的位置
    robot: Articulation = env.scene[asset_cfg.name]
    # 获取末端连杆的世界坐标和姿态
    link_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]
    link_quat_w = robot.data.body_quat_w[:, asset_cfg.body_ids[0]]
    # 将偏移量转换为张量并应用旋转
    offset_tensor = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(env.num_envs, 1)
    spoon_center_pos_w = link_pos_w + quat_apply(link_quat_w, offset_tensor)

    # 2. 获取小球在“世界坐标系”中的当前位置
    ball: RigidObject = env.scene[ball_cfg.name]
    # .data.root_pos_w 提供了小球在世界坐标系下的位置
    ball_pos_w = ball.data.root_pos_w

    # 3. 计算并返回小球与勺子中心之间的L2-norm（欧几里得距离）
    return torch.norm(ball_pos_w - spoon_center_pos_w, dim=1)




def position_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking of the position error using L2-norm.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame). The position error is computed as the L2-norm
    of the difference between the desired and current positions.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return torch.norm(curr_pos_w - des_pos_w, dim=1)


def position_command_error_tanh(
    env: ManagerBasedRLEnv, std: float, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
    """Reward tracking of the position using the tanh kernel.

    The function computes the position error between the desired position (from the command) and the
    current position of the asset's body (in world frame) and maps it with a tanh kernel.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current positions
    des_pos_b = command[:, :3]
    des_pos_w, _ = combine_frame_transforms(asset.data.root_pos_w, asset.data.root_quat_w, des_pos_b)
    curr_pos_w = asset.data.body_pos_w[:, asset_cfg.body_ids[0]]  # type: ignore
    distance = torch.norm(curr_pos_w - des_pos_w, dim=1)
    return 1 - torch.tanh(distance / std)


def orientation_command_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """Penalize tracking orientation error using shortest path.

    The function computes the orientation error between the desired orientation (from the command) and the
    current orientation of the asset's body (in world frame). The orientation error is computed as the shortest
    path between the desired and current orientations.
    """
    # extract the asset (to enable type hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    # obtain the desired and current orientations
    des_quat_b_xyzw = command[:, 3:7]
    des_quat_b = des_quat_b_xyzw[:, [3, 0, 1, 2]]  # convert to wxyz
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]  # type: ignore
    return quat_error_magnitude(curr_quat_w, des_quat_w)

def roll_pitch_orientation_error(env: ManagerBasedRLEnv, command_name: str, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    """
    【最终修正版 - 仅惩罚 Roll 和 Pitch】的姿态跟踪误差。

    此函数使用 euler_xyz_from_quat 将姿态转换为欧拉角，
    然后仅计算 roll 和 pitch 分量的误差。
    """
    # 提取机器人和指令
    asset: RigidObject = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # 获取目标和当前的姿态四元数 (w, x, y, z)
    des_quat_b_xyzw = command[:, 3:7]
    des_quat_b = des_quat_b_xyzw[:, [3, 0, 1, 2]]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]

    # 【核心修改】使用 euler_xyz_from_quat
    # 它返回一个元组 (roll, pitch, yaw)
    curr_roll, curr_pitch, _ = euler_xyz_from_quat(curr_quat_w)
    des_roll, des_pitch, _ = euler_xyz_from_quat(des_quat_w)

    # 将 roll 和 pitch 组合起来以便计算误差
    # .unsqueeze(1) 将 (N,) 变为 (N, 1)
    curr_rp = torch.cat((curr_roll.unsqueeze(1), curr_pitch.unsqueeze(1)), dim=1)
    des_rp = torch.cat((des_roll.unsqueeze(1), des_pitch.unsqueeze(1)), dim=1)

    # 计算 roll 和 pitch 两个维度上的欧几里得距离 (L2 范数)
    rp_error = torch.norm(curr_rp - des_rp, dim=1)

    return rp_error


def spoon_target_pose_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    offset: tuple[float, float, float],
    pos_threshold: float,
    rot_threshold: float,
) -> torch.Tensor:
    """
    【修正版】
    判断“勺子中心”和“Link6姿态”是否到达目标。
    """
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)

    # -- 1. 计算位置误差 (使用修正后的逻辑) --
    distance = spoon_position_command_error(env, command_name, asset_cfg, offset)
    pos_reached = distance < pos_threshold

    # -- 2. 计算姿态误差 (这部分逻辑不变，因为姿态是相对于Link6的) --
    des_quat_b_xyzw = command[:, 3:7]
    des_quat_b = des_quat_b_xyzw[:, [3, 0, 1, 2]]
    des_quat_w = quat_mul(asset.data.root_quat_w, des_quat_b)
    curr_quat_w = asset.data.body_quat_w[:, asset_cfg.body_ids[0]]
    rot_error = quat_error_magnitude(curr_quat_w, des_quat_w)
    rot_reached = rot_error < rot_threshold

    return pos_reached & rot_reached

def end_effector_workspace_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    forbidden_axis: str = "y",
    limit: float = -0.1,
    beta: float = 10.0
) -> torch.Tensor:
    """
    【最终修正版】当机械臂末端执行器进入不期望的工作空间时施加惩罚。
    此版本会计算相对于环境中心的局部坐标，解决了 env_spacing 导致的问题。
    """
    robot: Articulation = env.scene[asset_cfg.name]
    
    # 1. 获取末端执行器在世界坐标系中的绝对位置
    end_effector_pos_w = robot.data.body_pos_w[:, asset_cfg.body_ids[0]]

    # 2. 【关键修正】获取每个环境自身在世界坐标系中的原点位置
    env_origins = env.scene.env_origins

    # 3. 计算末端执行器相对于其环境原点的局部位置
    # end_effector_pos_local.shape: (num_envs, 3)
    end_effector_pos_local = end_effector_pos_w - env_origins


    if forbidden_axis == "x":
        coord = end_effector_pos_local[:, 0]
    elif forbidden_axis == "y":
        coord = end_effector_pos_local[:, 1]
    else: # "z"
        coord = end_effector_pos_local[:, 2]

    penetration_depth = -(coord - limit)
    penalty = torch.nn.functional.softplus(penetration_depth, beta=beta)
    
    return penalty



# ----- ▼▼▼ 新增的运送任务奖励函数 ▼▼▼ -----

def _get_spoon_center_position_w(env: ManagerBasedRLEnv, end_effector_cfg: SceneEntityCfg, offset: tuple[float, float, float]) -> torch.Tensor:
    """一个辅助函数，用于计算考虑偏移后的勺子中心在世界坐标系中的位置。"""
    robot: Articulation = env.scene[end_effector_cfg.name]
    
    # 获取 Link6 的世界坐标和姿态
    link_pos_w = robot.data.body_pos_w[:, end_effector_cfg.body_ids[0]]
    link_quat_w = robot.data.body_quat_w[:, end_effector_cfg.body_ids[0]] # 格式 (w, x, y, z)
    
    # 定义在 Link6 局部坐标系下的偏移向量
    offset_tensor = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(env.num_envs, 1)
    
    # 使用四元数旋转局部偏移向量，然后加到 Link6 的世界坐标上
    spoon_center_pos_w = link_pos_w + quat_apply(link_quat_w, offset_tensor)
    
    return spoon_center_pos_w

# 新增函数 1: 奖励球在勺子附近 (密集奖励)
def keep_ball_in_spoon_reward(
    env: ManagerBasedRLEnv,
    end_effector_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    offset: tuple[float, float, float],
    alpha: float = 10.0  # 控制奖励衰减速度的系数
) -> torch.Tensor:
    """当球离勺子中心很近时，提供一个密集的正奖励。"""
    spoon_center_pos_w = _get_spoon_center_position_w(env, end_effector_cfg, offset)
    ball: RigidObject = env.scene[ball_cfg.name]
    # -- 修正: 直接从数据缓冲区获取位置，性能更高且兼容新版API --
    ball_pos_w = ball.data.root_state_w[:, :3]

    # 计算球和勺子中心之间的距离
    distance = torch.norm(spoon_center_pos_w - ball_pos_w, dim=1)
    
    # 使用指数衰减函数作为奖励，距离越近，奖励越高，最高为1
    return torch.exp(-alpha * distance)

# 新增函数 2: 球掉落的惩罚 (稀疏惩罚)
def ball_dropped_penalty(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    offset: tuple[float, float, float],
    distance_threshold: float,
) -> torch.Tensor:
    """
    【修改后】如果球与勺子中心的距离超过阈值，则施加惩罚。
    这是一个基于距离的掉落判断标准。

    Args:
        env: 环境实例。
        asset_cfg: 机器人资产配置，用于定位勺子。
        ball_cfg: 小球资产配置。
        offset: 从末端连杆到勺子中心的偏移。
        distance_threshold: 判断为“掉落”的距离阈值。
    """
    # 计算勺子中心位置
    spoon_center_pos_w = _get_spoon_center_position_w(env, asset_cfg, offset)
    
    # 获取小球位置
    ball: RigidObject = env.scene[ball_cfg.name]
    ball_pos_w = ball.data.root_state_w[:, :3]
    
    # 计算球和勺子中心之间的距离
    distance = torch.norm(spoon_center_pos_w - ball_pos_w, dim=1)
    
    # 判断距离是否超过了掉落阈值
    is_dropped = distance > distance_threshold
    
    # 返回1.0代表掉落，0.0代表没有。权重将在配置文件中设为负数。
    return is_dropped.float()

# 新增函数 3: 判断球是否掉落 (用于终止条件)
def has_ball_dropped(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    offset: tuple[float, float, float],
    distance_threshold: float,
    grace_period_steps: int,    # 允许的宽限步数
) -> torch.Tensor:
    """
    【修改后】用于终止条件的函数，判断球是否因离勺子太远而“掉落”。
    返回布尔类型。
    """
    is_in_grace_period = env.episode_length_buf < grace_period_steps
    # 直接复用上面惩罚函数的逻辑，并将结果转换为布尔型
    return ball_dropped_penalty(env, asset_cfg, ball_cfg, offset, distance_threshold).bool()& ~is_in_grace_period.bool()


# ▼▼▼ 1. 内部逻辑函数 - 计算勺子和球与目标的接近程度 ▼▼▼
def _get_spoon_and_ball_proximity_info(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    offset: tuple[float, float, float], # 【物理偏移】: 勺子中心相对于末端连杆(link6)的固定物理偏移, 例如 (0.0, 0.0, 0.1185)
    pos_threshold: float,
    rot_threshold: float,
    reward_scaling: float,
) -> dict[str, torch.Tensor]:
    """
    【内部逻辑函数】
    根据精确的TCP位置计算勺子中心和球与目标(由command直接定义)的接近程度。
    """
    # -- 获取场景中的对象
    robot: Articulation = env.scene[asset_cfg.name]
    ball: RigidObject = env.scene[ball_cfg.name]
    command = env.command_manager.get_command(command_name)
    
    # -- 计算目标姿态 (世界坐标系)
    # 【修改点】: 目标姿态直接由 command 决定，不再添加任何额外偏移。
    des_pose_b = command # 假设 command 的格式为 [x, y, z, qw, qx, qy, qz]
    des_pos_b, des_rot_b_xyzw = des_pose_b[:, :3], des_pose_b[:, 3:7]

    des_rot_b = des_rot_b_xyzw[:, [3, 0, 1, 2]]  # 转换为 (w, x, y, z) 格式
    # 将基于机器人底座的目标，转换为世界坐标系的目标!!!!!!!!combine_frame_transforms需要传入(w, x, y, z) 格式的四元数

    des_pos_w, des_rot_w = combine_frame_transforms(
        robot.data.root_pos_w, robot.data.root_quat_w, des_pos_b, des_rot_b
    )
    
    # -- 计算勺子中心(TCP)的当前世界姿态 --
    robot_for_ee: Articulation = env.scene[asset_cfg.name]
    link_id = asset_cfg.body_ids[0]
    link_pos_w = robot_for_ee.data.body_pos_w[:, link_id]
    link_quat_w = robot_for_ee.data.body_quat_w[:, link_id]
    
    tcp_offset_tensor = torch.tensor(offset, device=env.device, dtype=torch.float).repeat(env.num_envs, 1)
    world_offset_vec = quat_apply(link_quat_w, tcp_offset_tensor)
    
    spoon_center_pos_w = link_pos_w + world_offset_vec
    spoon_center_rot_w = link_quat_w
    
    # -- 计算姿态误差 --
    spoon_pos_error = torch.norm(spoon_center_pos_w - des_pos_w, dim=1)
    # spoon_rot_error = quat_error_magnitude(spoon_center_rot_w, des_rot_w) # 你的正确修改
    

    # 【新代码】
    # 使用正确的函数将姿态四元数转换为欧拉角元组
    curr_roll, curr_pitch, _ = euler_xyz_from_quat(spoon_center_rot_w)
    des_roll, des_pitch, _ = euler_xyz_from_quat(des_rot_w)

    # 将 roll 和 pitch 组合起来
    curr_rp = torch.cat((curr_roll.unsqueeze(1), curr_pitch.unsqueeze(1)), dim=1)
    des_rp = torch.cat((des_roll.unsqueeze(1), des_pitch.unsqueeze(1)), dim=1)

    # 计算 roll 和 pitch 的综合误差
    spoon_rot_error = torch.norm(curr_rp - des_rp, dim=1)
    # ▲▲▲ 替换结束 ▲▲▲
    
    ball_pos_w = ball.data.root_state_w[:, :3]
    ball_pos_error = torch.norm(ball_pos_w - des_pos_w, dim=1)

    # -- 计算连续奖励 --
    # 总误差是勺子位置误差、勺子姿态误差 和 小球位置误差 的加权和
    total_error = spoon_pos_error + 0.8 * spoon_rot_error + ball_pos_error
    proximity_reward = torch.exp(-reward_scaling * total_error)

    # -- 判断是否成功（用于终止条件）--
    # 当勺子和小球都足够接近目标时，任务成功
    spoon_pos_reached = spoon_pos_error < pos_threshold
    spoon_rot_reached = spoon_rot_error < rot_threshold

    # 只打印第一个环境 (env 0) 的信息，避免刷屏
    if pos_threshold > 0 and rot_threshold > 0:  # 仅在阈值大于0时打印
        print(f"Current rot_error: {spoon_rot_error[0]:.4f}, rot_threshold: {rot_threshold}")
        print(f"Current pos_error: {spoon_pos_error[0]:.4f}, pos_threshold: {pos_threshold}")

        
    ball_reached = ball_pos_error < (pos_threshold + 0.02) # 球的成功半径可以稍大一些
    
    is_success = spoon_pos_reached & spoon_rot_reached & ball_reached

    return {"reward": proximity_reward, "is_success": is_success}


# ▼▼▼ 2. 最终包装器函数 - 用于奖励 ▼▼▼
def goal_and_ball_proximity_reward(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    offset: tuple[float, float, float], 
    reward_scaling: float,
) -> torch.Tensor:
    """【奖励函数 - 返回浮点数】"""
    info = _get_spoon_and_ball_proximity_info(
        env, command_name, asset_cfg, ball_cfg, offset, 0.0, 0.0, reward_scaling
    )
    return info["reward"]


# ▼▼▼ 3. 最终包装器函数 - 用于终止 ▼▼▼
def has_goal_and_ball_been_reached(
    env: ManagerBasedRLEnv,
    command_name: str,
    asset_cfg: SceneEntityCfg,
    ball_cfg: SceneEntityCfg,
    offset: tuple[float, float, float], 
    pos_threshold: float,
    rot_threshold: float,
) -> torch.Tensor:
    """【终止函数 - 返回布尔值】"""
    info = _get_spoon_and_ball_proximity_info(
        env, command_name, asset_cfg, ball_cfg, offset, pos_threshold, rot_threshold, 0.0
    )
    return info["is_success"]