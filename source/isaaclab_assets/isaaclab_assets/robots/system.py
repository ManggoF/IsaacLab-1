# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Configuration for the 皮带秤System.

The following configurations are available:

* :obj:`SYSTEM_CFG`: Franka Emika Panda robot with Panda hand
* :obj:`SYSTEM_HIGH_PD_CFG`: Franka Emika Panda robot with Panda hand with stiffer PD control

Reference: https://github.com/frankaemika/franka_ros
"""

import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR

##
# Configuration
##

SYSTEM_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/System/system.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=0
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.005, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            "joint1": -1.57,  # 连续旋转关节 (Continuous)
            "joint2": -0.3,  # 移动关节 (Prismatic), 范围: [-0.66, 0]
            "joint3": -0.02,  # 移动关节 (Prismatic), 范围: [-0.02, 0.02]
            "joint4": 0.02,  # 移动关节 (Prismatic), 范围: [-0.02, 0.02]
        },
    ),
    actuators={
        # TODO: 以下是执行器的示例配置。
        # 您需要根据您系统的实际电机参数（扭矩、速度、控制器刚度等）来调整这些值。
        "base_rotation_motor": ImplicitActuatorCfg(
            joint_names_expr=["joint1"],
            effort_limit_sim=100.0,    # 占位符: 最大力/扭矩 (N or Nm)
            velocity_limit_sim=5.0,    # 占位符: 最大速度 (m/s or rad/s)
            stiffness=200.0,           # 占位符: PD控制器中的P项 (刚度)
            damping=20.0,              # 占位符: PD控制器中的D项 (阻尼)
        ),
        "linear_slide_motor": ImplicitActuatorCfg(
            joint_names_expr=["joint2"],
            effort_limit_sim=300.0,    # 占位符: 最大力/扭矩 (N or Nm)
            velocity_limit_sim=1.0,    # 占位符: 最大速度 (m/s or rad/s)
            stiffness=800.0,           # 占位符: PD控制器中的P项 (刚度)
            damping=100.0,              # 占位符: PD控制器中的D项 (阻尼)
        ),
        "gripper_motors": ImplicitActuatorCfg(
            joint_names_expr=["joint[3-4]"], # 使用正则表达式匹配 joint3 和 joint4
            effort_limit_sim=30.0,     # 占位符: 最大力/扭矩 (N or Nm)
            velocity_limit_sim=0.5,    # 占位符: 最大速度 (m/s or rad/s)
            stiffness=100.0,           # 占位符: PD控制器中的P项 (刚度)
            damping=10.0,             # 占位符: PD控制器中的D项 (阻尼)
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)

# ===============================================================
#  高增益版配置 (可选，但推荐)
# ===============================================================

# 1. 从您的基础配置 CUSTOM_SYSTEM_CFG 创建一个副本
SYSTEM_HIGH_PD_CFG = SYSTEM_CFG.copy()
# 2. (可选) 在新配置中禁用重力
SYSTEM_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
# 3. 提高您自定义执行器的刚度和阻尼
#    注意：这里的名称 "base_rotation_motor", "linear_slide_motor", "gripper_motors"
#    必须与您在 CUSTOM_SYSTEM_CFG 中定义的执行器名称完全一致。
#    增加的倍数（例如5倍）是一个经验值，您可以根据仿真效果进行调整。

# 将旋转电机的刚度从 100 增加到 500
SYSTEM_HIGH_PD_CFG.actuators["base_rotation_motor"].stiffness = 500.0
SYSTEM_HIGH_PD_CFG.actuators["base_rotation_motor"].damping = 50.0 # 阻尼也应相应增加

# 将直线滑台电机的刚度从 400 增加到 2000
SYSTEM_HIGH_PD_CFG.actuators["linear_slide_motor"].stiffness = 2000.0
SYSTEM_HIGH_PD_CFG.actuators["linear_slide_motor"].damping = 100.0

# 将夹爪电机的刚度从 800 增加到 4000
SYSTEM_HIGH_PD_CFG.actuators["gripper_motors"].stiffness = 1000.0
SYSTEM_HIGH_PD_CFG.actuators["gripper_motors"].damping = 200.0

# 4. 为这个新配置添加说明文档
"""
Configuration for the custom mechatronic system with stiffer PD control.

This configuration increases the stiffness and damping gains of the actuators,
resulting in more precise and faster tracking of joint commands. It is particularly
useful for tasks requiring high-precision control, such as differential IK.

"""