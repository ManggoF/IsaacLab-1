# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.assets import RigidObjectCfg
# [修改 1]: 导入我们需要的Action配置类
from isaaclab.envs.mdp.actions import JointPositionActionCfg
from isaaclab.sensors import FrameTransformerCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import OffsetCfg
from isaaclab.sim.schemas.schemas_cfg import RigidBodyPropertiesCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from isaaclab_tasks.manager_based.manipulation.lift.lift_env_cfg import LiftEnvCfg

##
# Pre-defined configs
##
from isaaclab.markers.config import FRAME_MARKER_CFG  # isort: skip
# [修改 2]: 导入您自己的机器人配置，而不是Franka
from isaaclab_assets.robots.system import SYSTEM_CFG  # isort: skip


@configclass
class SystemCubeLiftEnvCfg(LiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # [修改 3]: 将机器人设置为您的自定义系统
        self.scene.robot = SYSTEM_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")

        
        self.actions.arm_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint1", "joint2"], # AI将直接控制这两个关节
            scale=0.5, # 将AI网络输出的[-1, 1]的值缩放到一个合理的关节运动范围
            use_default_offset=False, # 使用USD文件中定义的关节默认姿态作为中心点
            offset={
                    "joint1": -1.57,
                    "joint2": -0.3,
                },
        )


        # 4.2 连续控制夹爪动作 (这部分保持不变，它已经是正确的类型了)
        self.actions.gripper_action = JointPositionActionCfg(
            asset_name="robot",
            joint_names=["joint3", "joint4"],
            scale={"joint3": -0.01, "joint4": 0.01},
            offset={"joint3": -0.01, "joint4": 0.01},
            use_default_offset=False,
        )

        

        # [修改 5]: 将命令空间的目标物体名称更新为您的末端执行器连杆
        self.commands.object_pose.body_name = "Link2"

        # 设置要抓取的方块对象 (这部分无需修改)
        self.scene.object = RigidObjectCfg(
            prim_path="{ENV_REGEX_NS}/Object",
            init_state=RigidObjectCfg.InitialStateCfg(pos=[0, -0.05, 0.1], rot=[1, 0, 0, 0]),
            spawn=UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Shapes/cylinder.usd",
                scale=(1.5, 1.5, 1.5),
                rigid_props=RigidBodyPropertiesCfg(
                    # -- [核心修改 1] --
                    # 启用运动学模式。这将使物体忽略重力、摩擦力等。
                    kinematic_enabled=False,
                    # 以下参数保持不变
                    solver_position_iteration_count=16,
                    solver_velocity_iteration_count=1,
                    max_angular_velocity=1000.0,
                    max_linear_velocity=1000.0,
                    linear_damping=0.1,
                    angular_damping=0.1,
                    max_depenetration_velocity=5.0,
                    disable_gravity=False, # 即使这里是False, kinematic也会覆盖它
                ),
                
            ),
        )

        # [修改 6]: 更新FrameTransformer传感器以匹配您的机器人结构
        marker_cfg = FRAME_MARKER_CFG.copy()
        marker_cfg.markers["frame"].scale = (0.1, 0.1, 0.1)
        marker_cfg.prim_path = "/Visuals/FrameTransformer"
        self.scene.ee_frame = FrameTransformerCfg(
            # 6.1: 源坐标系 prim_path 从机器人基座开始 (base_link)
            prim_path="{ENV_REGEX_NS}/Robot/base_link",
            debug_vis=False,
            visualizer_cfg=marker_cfg,
            target_frames=[
                FrameTransformerCfg.FrameCfg(
                    # 6.2: 目标坐标系 prim_path 是您的末端连杆 (Link2)
                    prim_path="{ENV_REGEX_NS}/Robot/Link2",
                    name="end_effector", # 这个名字可以保持不变
                    # 6.3: offset 是从Link2原点到TCP（工具中心点）的偏移
                    offset=OffsetCfg(
                        pos=[-0.40285, 0.17497, 0.00561],
                    ),
                ),
            ],
        )

# [修改 7]: 更新Play配置的类名和继承关系
@configclass
class SystemCubeLiftEnvCfg_PLAY(SystemCubeLiftEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False