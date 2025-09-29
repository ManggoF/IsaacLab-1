# # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from dataclasses import MISSING

# import isaaclab.sim as sim_utils
# from isaaclab.sim.spawners.materials import RigidBodyMaterialCfg # 新增光滑冰面
# from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg

# from isaaclab.sim.spawners.shapes import CuboidCfg 
# from isaaclab.sim.schemas import RigidBodyPropertiesCfg, CollisionPropertiesCfg # 新增碰撞属性

# from isaaclab.envs import ManagerBasedRLEnvCfg
# from isaaclab.managers import CurriculumTermCfg as CurrTerm
# from isaaclab.managers import EventTermCfg as EventTerm
# from isaaclab.managers import ObservationGroupCfg as ObsGroup
# from isaaclab.managers import ObservationTermCfg as ObsTerm
# from isaaclab.managers import RewardTermCfg as RewTerm
# from isaaclab.managers import SceneEntityCfg
# from isaaclab.managers import TerminationTermCfg as DoneTerm
# from isaaclab.scene import InteractiveSceneCfg
# from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
# from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
# from isaaclab.utils import configclass
# from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

# from . import mdp

# ##
# # Scene definition
# ##


# @configclass
# class ObjectTableSceneCfg(InteractiveSceneCfg):
#     """Configuration for the lift scene with a robot and a object.
#     This is the abstract base implementation, the exact scene is defined in the derived classes
#     which need to set the target object, robot and end-effector frames
#     """

#     # robots: will be populated by agent env cfg
#     robot: ArticulationCfg = MISSING
#     # end-effector sensor: will be populated by agent env cfg
#     ee_frame: FrameTransformerCfg = MISSING
#     # target object: will be populated by agent env cfg
#     object: RigidObjectCfg | DeformableObjectCfg = MISSING

#     # 为每一个环境创建一个独立的、看不见的、光滑的刚体盒子
#     conveyor_surface = RigidObjectCfg(
#         # 关键1: 使用 {ENV_REGEX_NS}，确保每个环境都有自己的传送带
#         prim_path="{ENV_REGEX_NS}/ConveyorSurface",
        
#         # 关键2: 使用 RigidObjectCfg 专属的 InitialStateCfg
#         init_state=RigidObjectCfg.InitialStateCfg(pos=[0, 0, 0.03],lin_vel=[-0.6, 0.0, 0.0],), 
#         spawn=CuboidCfg(
#             size=(2, 1, 0.06),
#             rigid_props=RigidBodyPropertiesCfg(kinematic_enabled=False,disable_gravity=True),
#             collision_props=CollisionPropertiesCfg(collision_enabled=True),
#             physics_material=RigidBodyMaterialCfg(
#                 static_friction=0.8,
#                 dynamic_friction=0.5,
#                 restitution=0.0,
#             ),
#             # 关键3: 使用唯一正确的 "visible" 参数来实现隐形
#             visible=True,
#         ),
#     )

#     # Table
#     table = AssetBaseCfg(
#         prim_path="{ENV_REGEX_NS}/Table",
#         init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
#         spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable_system.usd"),
#     )

#     # plane
#     plane = AssetBaseCfg(
#         prim_path="/World/GroundPlane",
#         init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
#         spawn=GroundPlaneCfg(),
#     )

#     # lights
#     light = AssetBaseCfg(
#         prim_path="/World/light",
#         spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
#     )


# ##
# # MDP settings
# ##


# @configclass
# class CommandsCfg:
#     """Command terms for the MDP."""


#     object_pose = mdp.UniformPoseCommandCfg(
#         asset_name="robot",
#         body_name=MISSING,  # this will be set by agent env cfg
#         resampling_time_range=(5.0, 5.0),
#         debug_vis=True,
#         ranges=mdp.UniformPoseCommandCfg.Ranges(
#             # 将X和Y的目标范围固定在一个非常小的区域，这代表了机械臂正下方的“抓取线”
#             # 假设机械臂放置在(0.5, 0.0)
#             pos_x=(-1, -0.8),
#             pos_y=(-0.01, 0.01),
#             # Z是我们的目标提升高度，让它在一个范围内随机
#             pos_z=(0.3, 0.5),
#             # 旋转全部固定为0
#             roll=(0.0, 0.0),
#             pitch=(0.0, 0.0),
#             yaw=(0.0, 0.0),
#         ),
#     )


# @configclass
# class ActionsCfg:
#     """Action specifications for the MDP."""

#     # will be set by agent env cfg
#     arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
#     # gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
#     gripper_action: mdp.JointPositionActionCfg = MISSING


# @configclass
# class ObservationsCfg:
#     """Observation specifications for the MDP."""

#     @configclass
#     class PolicyCfg(ObsGroup):
#         """Observations for policy group."""

#         joint_pos = ObsTerm(func=mdp.joint_pos_rel)
#         joint_vel = ObsTerm(func=mdp.joint_vel_rel)
#         object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        
#         object_linear_velocity = ObsTerm(
#             func=mdp.root_lin_vel,  # <-- 调用我添加的函数
#             params={
#                 "asset_cfg": SceneEntityCfg("object"),
#                 "frame": "robot_root"  # <-- 在机器人坐标系下观测速度，让学习更容易
#             }
#         )

#         target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
#         actions = ObsTerm(func=mdp.last_action)

#         def __post_init__(self):
#             self.enable_corruption = True
#             self.concatenate_terms = True

#     # observation groups
#     policy: PolicyCfg = PolicyCfg()


# @configclass
# class EventCfg:
#     """Configuration for events."""

#     reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

#     reset_object_position = EventTerm(
#         func=mdp.reset_root_state_uniform,
#         mode="reset",
#         params={
#             "pose_range": {"x": (0, 0), "y": (0, 0), "z": (0.0, 0.0)},
#             "velocity_range":{"x": (-0.8, -0.8), "y": (0.0, 0.0), "z": (0.0, 0.0)},
#             "asset_cfg": SceneEntityCfg("object", body_names="Object"),
#         },
#     )
    


# @configclass
# class RewardsCfg:
#     """Reward terms for the MDP."""

#     # reaching_object = RewTerm(func=mdp.object_ee_distance, params={"std": 0.1}, weight=1.0)


#      # 1. 奖励水平对齐: 鼓励旋转关节1，让夹爪位于物块正上方
#     aligning_with_object = RewTerm(
#         func=mdp.horizontal_alignment,
#         params={"std": 0.2},
#         weight=2.0
#     )

#     # 2. 奖励垂直接近: 当对齐后，鼓励关节2下降以接近物块
#     approaching_object = RewTerm(
#         func=mdp.approach_object_vertically,
#         params={"std": 0.1, "horizontal_threshold": 0.05}, # 阈值设为5cm
#         weight=1.0
#     )

#     # 3. 奖励成功举起: 这个奖励仍然非常重要，是任务成功的关键信号
#     lifting_object = RewTerm(func=mdp.object_is_lifted, params={"minimal_height": 0.15}, weight=15.0)

#     # 4. 奖励跟踪目标高度: 这是最终要的奖励，引导智能体完成整个任务
#     #    它会隐式地要求智能体在正确的XY位置(由Command定义)抓住并举起到目标Z高度
#     object_goal_tracking = RewTerm(
#         func=mdp.object_goal_distance,
#         params={"std": 0.3, "minimal_height": 0.15, "command_name": "object_pose"},
#         weight=20.0, # 给予最高权重
#     )

#     # 5. 动作惩罚: 保留这些以使动作更平滑
#     action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1e-4)
#     joint_vel = RewTerm(
#         func=mdp.joint_vel_l2,
#         weight=-1e-4,
#         params={"asset_cfg": SceneEntityCfg("robot")},
#     )


# @configclass
# class TerminationsCfg:
#     """Termination terms for the MDP."""

#     time_out = DoneTerm(func=mdp.time_out, time_out=True)

#     object_dropping = DoneTerm(
#         func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
#     )


# @configclass
# class CurriculumCfg:
#     """Curriculum terms for the MDP."""

#     action_rate = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
#     )

#     joint_vel = CurrTerm(
#         func=mdp.modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
#     )


# ##
# # Environment configuration
# ##


# @configclass
# class LiftEnvCfg(ManagerBasedRLEnvCfg):
#     """Configuration for the lifting environment."""

#     # Scene settings
#     scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=4.5)
#     # Basic settings
#     observations: ObservationsCfg = ObservationsCfg()
#     actions: ActionsCfg = ActionsCfg()
#     commands: CommandsCfg = CommandsCfg()
#     # MDP settings
#     rewards: RewardsCfg = RewardsCfg()
#     terminations: TerminationsCfg = TerminationsCfg()
#     events: EventCfg = EventCfg()
#     curriculum: CurriculumCfg = CurriculumCfg()

#     def __post_init__(self):
#         """Post initialization."""
#         # general settings
#         self.decimation = 2
#         self.episode_length_s = 5.0
#         # simulation settings
#         self.sim.dt = 0.01  # 100Hz
#         self.sim.render_interval = self.decimation

#         self.sim.physx.bounce_threshold_velocity = 0.2
#         self.sim.physx.bounce_threshold_velocity = 0.01
#         self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
#         self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
#         self.sim.physx.friction_correlation_distance = 0.00625
