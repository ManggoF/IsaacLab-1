# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import DeformableObjectCfg
from isaaclab.controllers.differential_ik_cfg import DifferentialIKControllerCfg
from isaaclab.envs.mdp.actions.actions_cfg import DifferentialInverseKinematicsActionCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sim.spawners import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

import isaaclab_tasks.manager_based.manipulation.lift.mdp as mdp

from . import joint_pos_env_cfg

##
# Pre-defined configs
##
from isaaclab_assets.robots.ur5e import UR5E_HIGH_PD_CFG  # isort: skip


##
# Rigid object lift environment.
##


@configclass
class Ur5eCubeLiftDiffIKEnvCfg(joint_pos_env_cfg.Ur5eCubeLiftEnvCfg):
    """Configuration for lifting a cube with a UR5e+Robotiq robot using IK control."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 切换到高PD控制器配置，这对于IK跟踪至关重要
        self.scene.robot = UR5E_HIGH_PD_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 为机器人设置差分逆运动学(IK)动作
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
            controller=DifferentialIKControllerCfg(command_type="pose", use_relative_mode=False, ik_method="dls"),
            # 从wrist_3_link到夹爪TCP（工具中心点）的偏移
            body_offset=DifferentialInverseKinematicsActionCfg.OffsetCfg(pos=[0.0, 0.0, 0.14]),
        )


@configclass
class Ur5eCubeLiftDiffIKEnvCfg_PLAY(Ur5eCubeLiftDiffIKEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False


##
# Deformable object lift environment with Differential IK control.
##


@configclass
class Ur5eTeddyBearLiftEnvCfg(Ur5eCubeLiftDiffIKEnvCfg):
    """Configuration for lifting a teddy bear with a UR5e+Robotiq robot using IK control."""
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 将物体替换为可变形的泰迪熊
        self.scene.object = DeformableObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=DeformableObjectCfg.InitialStateCfg(pos=(0.5, 0, 0.05), rot=(0.707, 0, 0, 0.707)),
            spawn=UsdFileCfg(
                usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Objects/Teddy_Bear/teddy_bear.usd",
                scale=(0.01, 0.01, 0.01),
            ),
        )

        # 降低夹爪的刚度，以更“温柔”地对待泰迪熊
        # -- 将执行器名称从 "panda_hand" 修改为 "robotiq_gripper"
        self.scene.robot.actuators["robotiq_gripper"].effort_limit_sim = 50.0
        self.scene.robot.actuators["robotiq_gripper"].stiffness = 40.0
        self.scene.robot.actuators["robotiq_gripper"].damping = 10.0

        # 对于可变形物体，禁用物理复制
        # FIXME: This should be fixed by the PhysX replication system.
        self.scene.replicate_physics = False

        # 为可变形物体设置事件
        self.events.reset_object_position = EventTerm(
            func=mdp.reset_nodal_state_uniform,
            mode="reset",
            params={
                "position_range": {"x": (-0.1, 0.1), "y": (-0.25, 0.25), "z": (0.0, 0.0)},
                "velocity_range": {},
                "asset_cfg": SceneEntityCfg("object"),
            },
        )

        # 移除对于可变形物体计算成本高或不适用的奖励/终止项
        # self.terminations.object_dropping = None
        # self.rewards.reaching_object = None
        # self.rewards.lifting_object = None
        # self.rewards.object_goal_tracking = None
        # self.rewards.object_goal_tracking_fine_grained = None
        # self.observations.policy.object_position = None