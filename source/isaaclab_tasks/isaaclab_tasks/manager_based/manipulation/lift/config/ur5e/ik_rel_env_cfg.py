# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.utils import configclass

# -- 导入我们之前创建的、已适配UR5e的关节位置控制环境配置
from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.ur5e import UR5E_HIGH_PD_CFG  # isort: skip


@configclass
class Ur5eCubeLiftRelIKEnvCfg(joint_pos_env_cfg.Ur5eCubeLiftEnvCfg):
    """Configuration for lifting a cube with a UR5e+Robotiq robot using relative IK control."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 切换到高PD控制器配置，这对于IK跟踪至关重要
        self.scene.robot = UR5E_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 为机器人设置差分逆运动学(IK)动作，并使用相对模式
        self.actions.arm_action = DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            body_name="wrist_3_link", # IK控制的起始参考连杆
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=True, ik_method="dls"),
            scale=0.5, # 对输入的delta指令进行缩放
            # 从wrist_3_link到夹爪TCP（工具中心点）的偏移
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.14]),
        )


@configclass
class Ur5eCubeLiftRelIKEnvCfg_PLAY(Ur5eCubeLiftRelIKEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False