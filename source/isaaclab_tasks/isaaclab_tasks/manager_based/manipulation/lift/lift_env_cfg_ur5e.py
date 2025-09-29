# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, DeformableObjectCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors.frame_transformer.frame_transformer_cfg import FrameTransformerCfg
from isaaclab.sim.spawners.from_files.from_files_cfg import GroundPlaneCfg, UsdFileCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR

from . import mdp

##
# Scene definition
##


@configclass
class ObjectTableSceneCfg(InteractiveSceneCfg):
    """Configuration for the lift scene with a robot and a object.
    This is the abstract base implementation, the exact scene is defined in the derived classes
    which need to set the target object, robot and end-effector frames
    """

    # robots: will be populated by agent env cfg
    robot: ArticulationCfg = MISSING
    # end-effector sensor: will be populated by agent env cfg
    ee_frame: FrameTransformerCfg = MISSING
    # target object: will be populated by agent env cfg
    object: RigidObjectCfg | DeformableObjectCfg = MISSING

    # Table
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0.5, 0, 0], rot=[0.707, 0, 0, 0.707]),
        spawn=UsdFileCfg(usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd"),
    )

    # plane
    plane = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=[0, 0, -1.05]),
        spawn=GroundPlaneCfg(),
    )

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=3000.0),
    )


##
# MDP settings
##


@configclass
class CommandsCfg:
    """Command terms for the MDP."""

    # object_pose = mdp.UniformPoseCommandCfg(
    #     asset_name="robot",
    #     body_name=MISSING,  # will be set by agent env cfg
    #     resampling_time_range=(5.0, 5.0),
    #     debug_vis=True,
    #     ranges=mdp.UniformPoseCommandCfg.Ranges(
    #         pos_x=(0.4, 0.6), pos_y=(-0.25, 0.25), pos_z=(0.25, 0.5), roll=(0.0, 0.0), pitch=(0.0, 0.0), yaw=(0.0, 0.0)
    #     ),
    # )
    pass


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    # will be set by agent env cfg
    arm_action: mdp.JointPositionActionCfg | mdp.DifferentialInverseKinematicsActionCfg = MISSING
    gripper_action: mdp.BinaryJointPositionActionCfg = MISSING
    # gripper_action: mdp.JointPositionActionCfg = MISSING


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel = ObsTerm(func=mdp.joint_vel_rel)
        object_position = ObsTerm(func=mdp.object_position_in_robot_root_frame)
        # target_object_position = ObsTerm(func=mdp.generated_commands, params={"command_name": "object_pose"})
        # 现在，机器人也知道了它“应该去”的目标点在哪里 (相对于自己)
        # target_ee_position = ObsTerm(
        #     func=mdp.target_object_position_in_robot_root_frame,
        #     params={"target_offset": (0.0, 0.0, 0.0)} # 目标在圆柱体质心上方10cm
        # )
        actions = ObsTerm(func=mdp.last_action)

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    reset_object_position = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {"x": (0, 0), "y": (0, 0), "z": (0.0, 0.0)},
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("object", body_names="cylinder"),
        },
    )



@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

     # --- 核心奖励：平滑、有界地追踪目标 ---
    reaching_goal = RewTerm(
        func=mdp.reaching_goal_bounded, # << 使用我们新的、安全的函数
        weight=10.0,  # << 现在权重是正的，因为函数返回值在[0,1]之间
        params={
            "std": 0.2, # << 这是一个超参数，表示在20cm范围内奖励比较显著
        }
    )

    # 这是一个强烈的惩罚，告诉机器人“不要把手放得太低！”
    floor_collision_penalty = RewTerm(
        func=mdp.penalize_ee_below_height,
        weight=-100.0, # << 给予一个强大的负权重
        params={
            # 桌子的高度大约在0米，圆柱体高度约5.5cm。
            # 我们设定一个2cm的安全高度，低于这个高度就开始惩罚。
            "minimum_height": 0.02,
        }
    )
    # --- 辅助惩罚 ---
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)

    # --- 阶段 1: 水平对齐 (移动到物体正上方) ---
    # align_horizontally = RewTerm(
    #     func=mdp.horizontal_alignment,
    #     weight=1.5,#2.5
    #     params={"std": 0.1, "ee_frame_cfg": SceneEntityCfg("ee_frame")}
    # )

    # # --- 阶段 2: 垂直下降 (从正上方罩住物体) ---
    # # 这个奖励只有在水平对齐得很好时才会被激活
    # approach_vertically = RewTerm(
    #     func=mdp.approach_from_top,
    #     weight=2.5,#6.5
    #     params={"std": 0.1, "horizontal_threshold": 0.05, "ee_frame_cfg": SceneEntityCfg("ee_frame")}
    # )

    # --- 阶段 3: 成功抓取 ---
    # 我们保留原来的抓取奖励，但现在它是在对齐之后才更容易获得
    # grasping_object = RewTerm(
    #     func=mdp.grasp_success,
    #     weight=1, 
    #     params={
    #         # 新增的对齐判断阈值
    #         "horizontal_alignment_threshold": 0.02, # 5cm 以内算对齐
    #         "vertical_alignment_threshold": 0.04,   # 8cm 以内算对齐
    #         # 原有的参数
    #         "gripper_close_bonus": 15.0, # 成功抓取的额外奖励要大
    #         "object_ee_distance_threshold": 0.05, # 5cm 以内算抓到
    #         "gripper_joint_closed_threshold": 0.4, # 目标闭合值
    #         "robot_cfg": SceneEntityCfg("robot")
    #     }
    # )


    # --- 最终目标: 举起物体 ---
    # 这是最终目标，给予最高的奖励
    # lifting_object = RewTerm(
    #     func=mdp.object_is_lifted, 
    #     params={"minimal_height": 0.06}, 
    #     weight=25.0 # 给予最高权重，作为最终驱动力
    # )

    # --- 行为惩罚与平滑项 ---
    # 惩罚把物体撞飞的行为
    # object_vel = RewTerm(func=mdp.penalize_object_velocity, weight=-0.5)

    # 惩罚在预抓取位置闲逛
    # lingering = RewTerm(
    #     func=mdp.penalize_lingering,
    #     weight=0.5, # 权重不宜过大，起到一个“轻推”的作用
    #     params={"horizontal_threshold": 0.05, "vertical_threshold": 0.05}
    # )

    # # 保持动作平滑
    # action_rate = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
    # # 节约能源
    # joint_vel = RewTerm(
    #     func=mdp.joint_vel_l1,
    #     weight=-5e-2,
    #     params={"asset_cfg": SceneEntityCfg("robot")},
    # )



@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    object_dropping = DoneTerm(
        func=mdp.root_height_below_minimum, params={"minimum_height": -0.05, "asset_cfg": SceneEntityCfg("object")}
    )


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""

    # action_rate = CurrTerm(
    #     func=mdp.smooth_modify_reward_weight, params={"term_name": "action_rate", "weight": -1e-1, "num_steps": 10000}
    # )

    # joint_vel = CurrTerm(
    #     func=mdp.smooth_modify_reward_weight, params={"term_name": "joint_vel", "weight": -1e-1, "num_steps": 10000}
    # )

    # grasp_curr = CurrTerm(
    #     func=mdp.smooth_modify_reward_weight, params={"term_name": "grasping_object", "weight": 25, "num_steps": 10000}
    # )
    


##
# Environment configuration
##


@configclass
class LiftEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the lifting environment."""

    # Scene settings
    scene: ObjectTableSceneCfg = ObjectTableSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    # commands: CommandsCfg = CommandsCfg()
    commands: CommandsCfg = None
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.episode_length_s = 5.0
        # simulation settings
        self.sim.dt = 0.01  # 100Hz
        self.sim.render_interval = self.decimation

        self.sim.physx.bounce_threshold_velocity = 0.2
        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.gpu_found_lost_aggregate_pairs_capacity = 1024 * 1024 * 4
        self.sim.physx.gpu_total_aggregate_pairs_capacity = 16 * 1024
        self.sim.physx.friction_correlation_distance = 0.00625
