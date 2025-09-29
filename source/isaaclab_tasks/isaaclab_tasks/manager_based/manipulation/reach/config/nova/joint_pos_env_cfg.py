# # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# import math

# from isaaclab.utils import configclass

# import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
# from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

# ##
# # Pre-defined configs
# ##
# from isaaclab_assets import NOVA_CFG  # isort: skip


# ##
# # Environment configuration
# ##


# @configclass
# class NovaReachEnvCfg(ReachEnvCfg):
#     def __post_init__(self):
#         super().__post_init__()

#         # 1. 切换机器人
#         self.scene.robot = NOVA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

#         # 2. 把末端连杆改成 Link6（或 USD 里实际名字）
#         ee_body = "Link6"
#         self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [ee_body]
#         self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [ee_body]
#         self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [ee_body]
       

#         # 为新的成功奖励指定末端连杆
#         self.rewards.goal_bonus.params["asset_cfg"].body_names = [ee_body]
#         # # 为新的成功终止条件指定末端连杆
#         # self.terminations.goal_reached.params["asset_cfg"].body_names = [ee_body]
#         # # -- 添加结束 --

#         # 3. 关节名用 URDF 里的 joint1~6
#         self.actions.arm_action = mdp.JointPositionActionCfg(
#             asset_name="robot",
#             joint_names=["joint[1-6]"],      # 正则匹配 1~6
#             scale=0.5,
#             use_default_offset= True,  
#         )

#         # 4. 末端指令也指向 Link6
#         self.commands.ee_pose.body_name = ee_body


# @configclass
# class NovaReachEnvCfg_PLAY(NovaReachEnvCfg):
#     def __post_init__(self):
#         # post init of parent
#         super().__post_init__()
#         # make a smaller scene for play
#         self.scene.num_envs = 50
#         self.scene.env_spacing = 2.5
#         # disable randomization for play
#         self.observations.policy.enable_corruption = False

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import math

from isaaclab.utils import configclass

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
# ▼▼▼ 导入我们新的 actions.py 文件 ▼▼▼
from isaaclab_tasks.manager_based.manipulation.reach.mdp import cus_actions as mdp_actions
from isaaclab_tasks.manager_based.manipulation.reach.reach_env_cfg import ReachEnvCfg

##
# Pre-defined configs
##
from isaaclab_assets import NOVA_CFG  # isort: skip


##
# Environment configuration
##


@configclass
class NovaReachEnvCfg(ReachEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        # 1. 切换机器人
        self.scene.robot = NOVA_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 2. 把末端连杆改成 Link6
        ee_body = "Link6"
        self.rewards.end_effector_position_tracking.params["asset_cfg"].body_names = [ee_body]
        self.rewards.end_effector_position_tracking_fine_grained.params["asset_cfg"].body_names = [ee_body]
        self.rewards.end_effector_orientation_tracking.params["asset_cfg"].body_names = [ee_body]
        # self.rewards.ball_position_tracking.params["asset_cfg"].body_names = [ee_body]
        # self.rewards.ball_dropped.params["asset_cfg"].body_names = [ee_body]
        self.rewards.workspace_penalty.params["asset_cfg"].body_names = [ee_body]
        self.rewards.workspace_penalty_z.params["asset_cfg"].body_names = [ee_body]

        # -- 运球任务奖励 --
        self.rewards.keep_ball_in_spoon.params["end_effector_cfg"].body_names = [ee_body]
        self.rewards.task_reward.params["asset_cfg"].body_names = [ee_body]
        # self.rewards.goal_bonus.params["asset_cfg"].body_names = [ee_body]

        # -- 终止条件 -- 
        self.terminations.goal_reached.params["asset_cfg"].body_names = [ee_body]
        self.terminations.ball_dropped.params["asset_cfg"].body_names = [ee_body]
     
        # -- 观察空间 --
        self.observations.policy.ball_position.params["ee_asset_cfg"].body_names = [ee_body]
       

        # 3. ▼▼▼ 使用新的带延迟的动作 ▼▼▼
        self.actions.arm_action = mdp_actions.DelayedJointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint[1-6]"],
            scale=0.5,
            use_default_offset=True,
            delay_s=1.0,  # 在前 1.0 秒内保持静止
        )

        # 4. 末端指令也指向 Link6
        # 注意：这里的指令现在只作为目标提供给奖励函数和观察，
        # 机械臂的实际行为在开始时由 DelayedJointPositionAction 控制
        self.commands.ee_pose.body_name = ee_body


@configclass
class NovaReachEnvCfg_PLAY(NovaReachEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
