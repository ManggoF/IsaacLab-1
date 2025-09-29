# 文件名: custom_pdp.py

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from . import mdp

"""
-----------------------------
-- 基础/工具类奖励函数 --
-----------------------------
"""

def object_to_stack_target_distance(
    env: ManagerBasedRLEnv, upper_object_cfg: SceneEntityCfg, lower_object_cfg: SceneEntityCfg, *, top_clearance: float
):
    upper_pos = env.scene[upper_object_cfg.name].data.root_pos_w
    lower_pos = env.scene[lower_object_cfg.name].data.root_pos_w
    target = lower_pos.clone()
    target[:, 2] += top_clearance
    dist = torch.norm(upper_pos - target, dim=-1)
    return -dist

def cube_is_grasped_and_lifted(
    env: ManagerBasedRLEnv, *, robot_cfg: SceneEntityCfg, ee_frame_cfg: SceneEntityCfg, object_cfg: SceneEntityCfg, lift_h: float
):
    grasped = mdp.object_grasped(env, robot_cfg=robot_cfg, ee_frame_cfg=ee_frame_cfg, object_cfg=object_cfg)
    obj_h = env.scene[object_cfg.name].data.root_pos_w[:, 2]
    lifted = obj_h > lift_h
    return (grasped & lifted).to(obj_h.dtype)

# -- 新增函数：惩罚末端执行器高度过低 --
def ee_height_penalty(env: ManagerBasedRLEnv, ee_frame_cfg: SceneEntityCfg, *, min_height: float, penalty_weight: float):
    """如果末端执行器低于指定高度，则给予惩罚。"""
    ee_z_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, 2]
    is_too_low = (ee_z_pos < min_height).float()
    # 惩罚值 = 权重 * 低于高度的距离
    penalty = penalty_weight * (min_height - ee_z_pos) * is_too_low
    return -penalty # 返回负值作为惩罚

"""
-----------------------------
-- 核心稠密塑形奖励函数 (全新逻辑: 统一惩罚) --
-----------------------------
"""

def phased_stacking_reward(
    env: ManagerBasedRLEnv,
    *,
    robot_cfg: SceneEntityCfg,
    ee_frame_cfg: SceneEntityCfg,
    object_cfg: SceneEntityCfg,
    lower_object_cfg: SceneEntityCfg,
    # 权重
    xy_dist_weight: float,
    z_dist_weight: float,
    gripper_penalty_weight: float,
    stack_dist_weight: float,
    # 几何参数
    lift_h: float,
    hover_height: float,
    xy_align_thresh: float,
    top_clearance: float,
):
    """
    一个完全基于负向惩罚的、分阶段的稠密奖励函数。
    """
    # 获取数据
    robot = env.scene[robot_cfg.name]
    ee_pos = env.scene[ee_frame_cfg.name].data.target_pos_w[:, 0, :]
    obj_pos = env.scene[object_cfg.name].data.root_pos_w
    
    # 手动计算夹爪宽度
    joint_names_list = robot.joint_names
    finger1_idx = joint_names_list.index("panda_finger_joint1")
    finger2_idx = joint_names_list.index("panda_finger_joint2")
    all_joint_pos = robot.data.joint_pos
    gripper_width = all_joint_pos[:, finger1_idx] + all_joint_pos[:, finger2_idx]

    # --- 条件门 ---
    is_lifted_gate = cube_is_grasped_and_lifted(
        env,
        robot_cfg=robot_cfg,
        ee_frame_cfg=ee_frame_cfg,
        object_cfg=object_cfg,
        lift_h=lift_h
    )

    # --- 抓取前惩罚 (Pre-grasp Penalties) ---
    
    # 惩罚1: XY平面距离惩罚
    dist_xy = torch.norm(ee_pos[:, :2] - obj_pos[:, :2], dim=-1)
    penalty_xy = xy_dist_weight * dist_xy

    # 惩罚2: Z轴距离惩罚 (带门控)
    # 目标：当XY未对齐时，保持在悬停高度；对齐后，下降到物体高度
    hover_target_z = obj_pos[:, 2] + hover_height
    grasp_target_z = obj_pos[:, 2] + 0.01  # 抓取时目标稍高于物体中心
    
    is_xy_aligned = (dist_xy < xy_align_thresh)
    target_z = torch.where(is_xy_aligned, grasp_target_z, hover_target_z)
    dist_z = torch.abs(ee_pos[:, 2] - target_z)
    penalty_z = z_dist_weight * dist_z
    
    # 惩罚3: 夹爪张开惩罚 (带门控)
    # 当靠近物体时，如果夹爪还是张开的，就给予惩罚
    is_near_object = is_xy_aligned & (dist_z < 0.05)
    # gripper_width > 0.01 表示夹爪是张开的
    penalty_gripper = gripper_penalty_weight * (gripper_width * is_near_object.float())
    
    total_pre_grasp_penalty = -(penalty_xy + penalty_z + penalty_gripper)

    # --- 抓取后惩罚 (Post-grasp Penalty) ---
    dist_to_stack_target = object_to_stack_target_distance(
        env,
        upper_object_cfg=object_cfg,
        lower_object_cfg=lower_object_cfg,
        top_clearance=top_clearance
    )
    total_post_grasp_penalty = stack_dist_weight * dist_to_stack_target

    # --- 根据总条件门选择最终奖励 ---
    final_reward = torch.where(is_lifted_gate.bool(), total_post_grasp_penalty, total_pre_grasp_penalty)
    return final_reward