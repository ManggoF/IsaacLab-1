# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the Universal Robots UR5e.

The following configurations are available:

* :obj:`UR5E_CFG`: Universal Robots UR5e with Robotiq 2F-85 gripper.
* :obj:`UR5E_HIGH_PD_CFG`: Universal Robots UR5e with Robotiq 2F-85 gripper with stiffer PD control.

"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

UR5E_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/UniversalRobots/ur5e_2f85.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=16, solver_velocity_iteration_count=2
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
    joint_pos={
        "shoulder_pan_joint": 0.0,
        "shoulder_lift_joint": -1.2,  # 抬起一点
        "elbow_joint": 1.57,        # 弯曲手肘
        "wrist_1_joint": -2.355,      # 调整手腕姿态
        "wrist_2_joint": -1.57,       # 让手腕朝下
        "wrist_3_joint": 0.0,
        "finger_joint": 0.785,        # 夹爪张开
    },
),
     actuators={
        # -->>> 将手臂关节分为两组，更符合物理现实
        "ur5e_base_joints": ImplicitActuatorCfg(
            # 匹配 shoulder_pan_joint, shoulder_lift_joint, elbow_joint
            joint_names_expr=["(shoulder.*|elbow)_joint"],
            effort_limit_sim=150.0,  # 官方峰值扭矩
            stiffness=100.0,       # 基础刚度
            damping=10.0,
        ),
        "ur5e_wrist_joints": ImplicitActuatorCfg(
            # 匹配 wrist_1_joint, wrist_2_joint, wrist_3_joint
            joint_names_expr=["wrist.*_joint"],
            effort_limit_sim=28.0,   # 官方峰值扭矩
            stiffness=100.0,      # 基础刚度
            damping=10.0,
        ),
        "robotiq_gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=100.0,
            stiffness=1000.0,
            damping=50.0,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of Universal Robots UR5e robot with Robotiq 2F-85 gripper."""


UR5E_HIGH_PD_CFG = UR5E_CFG.copy()
UR5E_HIGH_PD_CFG.actuators["ur5e_base_joints"].stiffness = 1000.0
UR5E_HIGH_PD_CFG.actuators["ur5e_base_joints"].damping = 80.0
UR5E_HIGH_PD_CFG.actuators["ur5e_wrist_joints"].stiffness = 400.0
UR5E_HIGH_PD_CFG.actuators["ur5e_wrist_joints"].damping = 40.0
"""Configuration of Universal Robots UR5e robot with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""