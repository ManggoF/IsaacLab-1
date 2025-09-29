# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""
Configuration for the Universal Robots UR5e with Robotiq 2F-85 Gripper.

The following configurations are available:

* :obj:`UR5E_CFG`: UR5e robot with Robotiq 2F-85 gripper.
* :obj:`UR5E_HIGH_PD_CFG`: UR5e robot with Robotiq 2F-85 gripper with stiffer PD control.
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
            enabled_self_collisions=True, 
            solver_position_iteration_count=8, 
            solver_velocity_iteration_count=0
        ),
    ),
    

    # init_state=ArticulationCfg.InitialStateCfg(
    #     joint_pos={
    #         # Arm
    #         "shoulder_pan_joint": 0.0,
    #         "shoulder_lift_joint": -1.35,
    #         "elbow_joint": 1.74,
    #         "wrist_1_joint": -1.96,
    #         "wrist_2_joint": -1.57,
    #         "wrist_3_joint": 0.0,
    #         # Gripper (assuming 0 is open)
    #         "finger_joint": 0.0,
    #     },
    # ),
    init_state=ArticulationCfg.InitialStateCfg(
        joint_pos={
            # <<<<<<<<<<<< 核心修改：这是一个更舒展、面向前方的姿态 >>>>>>>>>>>>
            "shoulder_pan_joint": 0.0,
            "shoulder_lift_joint": -1.57, # -90度，手臂向前平举
            "elbow_joint": 1.57,          # +90度，形成一个直角
            "wrist_1_joint": -1.57,       # -90度
            "wrist_2_joint": -1.57,       # -90度，让夹爪朝下
            "wrist_3_joint": 0.0,
            # Gripper (保持张开)
            "finger_joint": 0.0,
        },
    ),
    actuators={
        "ur5e_shoulder_elbow": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint"],
            effort_limit_sim=150.0,  # Based on UR5e specs for base/shoulder joints
            velocity_limit_sim=3.14, # 180 deg/s
            stiffness=120.0,
            damping=10.0,
        ),
        "ur5e_wrist": ImplicitActuatorCfg(
            joint_names_expr=["wrist_1_joint", "wrist_2_joint", "wrist_3_joint"],
            effort_limit_sim=28.0,   # Based on UR5e specs for wrist joints
            velocity_limit_sim=3.14, # 180 deg/s
            stiffness=100.0,
            damping=10.0,
        ),
        "robotiq_gripper": ImplicitActuatorCfg(
            joint_names_expr=["finger_joint"],
            effort_limit_sim=100.0, # Adjusted for gripper
            velocity_limit_sim=0.8, # Adjusted for gripper
            stiffness=300.0,
            damping=30.0,
        ),
    },
    
    soft_joint_pos_limit_factor=1.0,
)
"""Configuration of UR5e robot with Robotiq 2F-85 gripper."""


UR5E_HIGH_PD_CFG = UR5E_CFG.copy()
UR5E_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
UR5E_HIGH_PD_CFG.actuators["ur5e_shoulder_elbow"].stiffness = 400.0
UR5E_HIGH_PD_CFG.actuators["ur5e_shoulder_elbow"].damping = 40.0
UR5E_HIGH_PD_CFG.actuators["ur5e_wrist"].stiffness = 400.0
UR5E_HIGH_PD_CFG.actuators["ur5e_wrist"].damping = 40.0
"""Configuration of UR5e robot with Robotiq 2F-85 gripper with stiffer PD control.

This configuration is useful for task-space control using differential IK.
"""