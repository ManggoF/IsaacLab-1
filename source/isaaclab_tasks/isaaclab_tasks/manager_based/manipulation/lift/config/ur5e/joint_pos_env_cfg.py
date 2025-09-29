# Copyright (c) 2022-2025, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift import mdp
from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg_ur5e import LiftEnvCfg
from isaaclab.envs.mdp.actions import DifferentialInverseKinematicsActionCfg as IKActionCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# -- 从我们上一步创建的文件中导入UR5e+Robotiq的配置
from isaaclab_assets.robots.ur5e import UR5E_CFG  # isort: skip


@configclass
class Ur5eCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # 设置UR5e+Robotiq作为机器人
        self.scene.robot = UR5E_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        # 为UR5e手臂设置动作
        # self.actions.arm_action = mdp.JointPositionActionCfg(
        #     asset_name="robot",
        #     joint_names=[
        #         "shoulder_pan_joint",
        #         "shoulder_lift_joint",
        #         "elbow_joint",
        #         "wrist_1_joint",
        #         "wrist_2_joint",
        #         "wrist_3_joint",
        #     ],
        #     scale=0.5,
        #     use_default_offset=True,
        # )
         # 监听并可视化末端执行器(TCP)的变换
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            prim_path="{ENV_REGEX_NS}/Robot/ur5e/base_link", # 从UR5e的基座开始计算
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    prim_path="{ENV_REGEX_NS}/Robot/ur5e/wrist_3_link", # 目标是UR5e的末端连杆
                    name="end_effector",
                    offset=OffsetCfg(
                        # NOTE: 这是从wrist_3_link到Robotiq夹爪抓取中心的估计偏移，您可能需要微调
                        pos=[0.0, 0.0, 0.14],
                        rot=[1.0, 0.0, 0.0, 0.0],  # 无旋转偏移
                    ),
                ),
            ],
        )
          # <<<<<<<<<<< 全新的、正确的逆运动学(IK)控制定义 >>>>>>>>>>>

       # 1. 首先，严格按照定义来配置IK控制器
        ik_controller = mdp.DifferentialIKControllerCfg(
            command_type="pose",        # << 控制完整位姿 (位置+方向)
            use_relative_mode=True,     # << 使用相对模式 (发送 delta pose)
            ik_method="dls",            # << 使用Damped Least Squares方法
        )

        # 2. 然后，配置IK动作本身，并将控制器和TCP偏移量传入
        self.actions.arm_action = mdp.DifferentialInverseKinematicsActionCfg(
            asset_name="robot",
            # 'joint_names' 是IK求解器可以控制的关节
            joint_names=[
                "shoulder_pan_joint",
                "shoulder_lift_joint",
                "elbow_joint",
                "wrist_1_joint",
                "wrist_2_joint",
                "wrist_3_joint",
            ],
            # 'body_name' 是我们名义上控制的目标连杆
            body_name="wrist_3_link",
            # 'body_offset' 才是设置TCP偏移的正确位置
            body_offset=IKActionCfg.OffsetCfg(
                pos=self.scene.ee_frame.target_frames[0].offset.pos,
                rot=self.scene.ee_frame.target_frames[0].offset.rot,
            ),
            # 'controller' 是我们刚刚定义的IK控制器实例
            controller=ik_controller,
            # 'scale' 依然是 (x, y, z, roll, pitch, yaw) 的速度缩放
            scale=(0.1, 0.1, 0.1, 0.5, 0.5, 0.5),
        )

        # 为Robotiq夹爪设置动作
        self.actions.gripper_action = mdp.BinaryJointPositionActionCfg(
            asset_name="robot",
            joint_names=["finger_joint"],
            # 0.0 表示完全张开
            open_command_expr={"finger_joint": 0.0},
            # 0.8 弧度表示闭合 (可根据需要微调)
            close_command_expr={"finger_joint": 0.8},
        )
        # 将末端执行器的身体名称设置为UR5e的末端连杆
        # self.commands.object_pose.body_name = "wrist_3_link"

        # 设置Cube作为要抓取的物体
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0.5, 0, 0.055], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cylinder_instance.usd",
                scale=(0.1, 0.1, 0.1),
                rigid_props=RigidBodyPropertiesCfg(
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False,
                ),
            ),
        )

       


@configclass
class Ur5eCubeLiftEnvCfg_PLAY(Ur5eCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False