# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from dataclasses import MISSING
import math

import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.managers import ActionTermCfg as ActionTerm
from isaaclab.managers import CurriculumTermCfg as CurrTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.utils.noise import AdditiveUniformNoiseCfg as Unoise

import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp
# 添加这行，导入你的自定义指令
from .mdp import cus_command as mdp_custom
##
# Scene definition
##


@configclass
class ReachSceneCfg(InteractiveSceneCfg):
    """Configuration for the scene with a robotic arm."""

    # world
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Mounts/SeattleLabTable/table_instanceable.usd",
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.55, 0.0, 0.0), rot=(0.70711, 0.0, 0.0, 0.70711)),
    )

    # 球
    ball = RigidObjectCfg(
    prim_path="{ENV_REGEX_NS}/Ball",
    spawn=sim_utils.SphereCfg(
        radius=0.01,  # 半径 1 cm
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,  # 启用重力
            kinematic_enabled=False,    # 物理模拟启用
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
                collision_enabled=True,
                rest_offset=0.0,
                contact_offset=0.001 # 增加接触偏移量，帮助物理引擎提前检测碰撞
            ),
        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=(1.0, 0.0, 0.0)),
        mass_props=sim_utils.MassPropertiesCfg(
            density=1000.0, # 比水稍大密度
        ),
        physics_material=sim_utils.RigidBodyMaterialCfg(
            static_friction=0.8,    # 静摩擦系数适合勺子摩擦
            dynamic_friction=0.8,   # 动摩擦系数适合勺子摩擦
            restitution=0.1,    # 低弹性，防止乱弹
        ),
    ),
    init_state=RigidObjectCfg.InitialStateCfg(
        pos=(0.2075, -0.67538, 0.6),   # 放在勺子上方，和你的 start_pose 对齐
    ),
)

    # robots
    robot: ArticulationCfg = MISSING

    # lights
    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(color=(0.75, 0.75, 0.75), intensity=2500.0),
    )



@configclass
class CommandsCfg:
    """MDP的指令项配置。"""

    # 使用恒定目标姿态指令
    ee_pose = mdp_custom.ConstantPoseCommandCfg(
        asset_name="robot",
        debug_vis=True,
        
        # 定义唯一的目标姿态
        target_pose=mdp_custom.ConstantPoseCommandCfg.Pose(
            pos_x=0.5,  #0.3
            pos_y=0.4,  #0.6
            pos_z=0.3,  #0.4
            roll=-math.pi / 2,  #-math.pi / 2
            pitch=0,
            yaw=0,  #-math.pi / 2
        ),
    )


@configclass
class ActionsCfg:
    """Action specifications for the MDP."""

    arm_action: ActionTerm = MISSING
    gripper_action: ActionTerm | None = None
    


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        # observation terms (order preserved)
        joint_pos = ObsTerm(func=mdp.joint_pos_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        joint_vel = ObsTerm(func=mdp.joint_vel_rel, noise=Unoise(n_min=-0.01, n_max=0.01))
        pose_command = ObsTerm(func=mdp.generated_commands, params={"command_name": "ee_pose"})
         # ▼▼▼ 关键新增: 告诉机器人球在哪里 ▼▼▼
        # 这个函数计算球相对于末端执行器的位置
        ball_position = ObsTerm(func=mdp.object_ee_position, 
                                params={"asset_cfg": SceneEntityCfg("ball"), 
                                        "ee_asset_cfg": SceneEntityCfg("robot", body_names=MISSING)})
        actions = ObsTerm(func=mdp.last_action)
        
        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = True

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class EventCfg:
    """Configuration for events."""

    reset_robot_joints = EventTerm(
        func=mdp.reset_joints_by_scale,
        mode="reset",
        params={
            "position_range": (1.0, 1.0),   # (0.5, 1.5)
            "velocity_range": (0.0, 0.0),
        },
    )

    reset_ball_position = EventTerm(
        func=mdp.reset_root_state_uniform,  # 使用这个函数来重置刚体的位置和速度
        mode="reset",  # 确保这个事件在每个回合开始时触发
        params={
            "asset_cfg": SceneEntityCfg("ball"),  # 指定要重置的物体是 "ball"
            # 将重置位置的范围设为一个固定点 (最小值和最大值相同)
            # 这个位置为在 ball 设置的 init_state.pos 基础上的偏移量
            "pose_range": {"x": (0, 0), "y": (0, 0), "z": (0, 0)},
            # 将速度重置为零
            "velocity_range": {"x": (0.0, 0.0), "y": (0.0, 0.0), "z": (0.0, 0.0)},
        }
    )


LINK6_SPOON_OFFSET = (0.0, 0.0, 0.1185)  # Link6 Z轴方向偏移11.9cm
@configclass
class RewardsCfg:
    """Reward terms for the MDP."""

    #  1. 末端执行器位置和方向追踪
    end_effector_position_tracking = RewTerm(
        func=mdp.spoon_position_command_error,
        weight=-5,    # -0.3
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose", "offset": LINK6_SPOON_OFFSET},
    )

    end_effector_position_tracking_fine_grained = RewTerm(
        func=mdp.spoon_position_command_error_tanh,
        weight=10, # 0.5
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "std": 0.5, "command_name": "ee_pose", "offset": LINK6_SPOON_OFFSET},
    )
    
    end_effector_orientation_tracking = RewTerm(
        func=mdp.roll_pitch_orientation_error,  # << 确保是这个新函数
        weight=-5, # 您可以根据需要调整权重
        params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    )

    # end_effector_orientation_tracking = RewTerm(
    #     func=mdp.orientation_command_error,
    #     weight=-5,    # -0.2
    #     params={"asset_cfg": SceneEntityCfg("robot", body_names=MISSING), "command_name": "ee_pose"},
    # )


    # 2. 球位置追踪与掉落惩罚 TODO：后面课程学习，让这个奖励变小一点，防止一直保持不动

    keep_ball_in_spoon = RewTerm(
        func=mdp.keep_ball_in_spoon_reward,
        weight=20.0,  # 给予一个不错的正奖励来鼓励保持球在勺子上 1.5
        params={
            "end_effector_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "ball_cfg": SceneEntityCfg("ball"),
            "offset": LINK6_SPOON_OFFSET,
        },
    )


    task_reward= RewTerm(
        func=mdp.goal_and_ball_proximity_reward,
        weight=200.0,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "ball_cfg": SceneEntityCfg("ball"),
            "offset": LINK6_SPOON_OFFSET, 
            "reward_scaling": 2.25,  # 奖励缩放因子
    }
)


    
    #     # ▼▼▼ 运送任务的核心奖励与惩罚 ▼▼▼

    # progress_to_goal = RewTerm(
    #     func=mdp.progress_to_goal_reward,   #type: ignore 构造一个类fun作为势函数
    #     weight=0, # << 初始权重很低，早期不鼓励移动 3.0
    #     params={"command_name": "ee_pose", "ball_cfg": SceneEntityCfg("ball")}
    # )   

    # ball_press_penalty = RewTerm(
    #     func=mdp.ball_too_close_to_table_penalty,
    #     weight=-3.0, # 给予一个强烈的负面信号
    #     params={
    #         "ball_cfg": SceneEntityCfg("ball"),
    #         "table_height": 0.0,  # 假设桌面在 z=0 平面
    #         "min_height_threshold": 0.02, # 当球离桌面低于2cm时开始惩罚
    #     },
    # )

    # action penalty
    action_rate = RewTerm(func=mdp.action_rate_l2, weight=-1)  #-0.0001
    joint_vel = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-3, #-0.0001
        params={"asset_cfg": SceneEntityCfg("robot")},
    )
    workspace_penalty = RewTerm(
        func=mdp.end_effector_workspace_penalty,
        weight=-20.0,  # << 给予一个强大的负权重
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "forbidden_axis": "x",
            "limit": -0.1,  # 桌子前方的x轴通常是正的，我们惩罚x变为负数
        },
    )
    workspace_penalty_z = RewTerm(
        func=mdp.end_effector_workspace_penalty,
        weight=-100.0,  # << 给予一个强大的负权重
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "forbidden_axis": "z",
            "limit": 0.15,  # 桌子前方的x轴通常是正的，我们惩罚x变为负数
        },
    )

@configclass
class TerminationsCfg:
    """Termination terms for the MDP."""
    # 任务完成 (位置和姿态都接近目标就算完成)

    # goal_reached = DoneTerm(
    #     func=mdp.has_goal_and_ball_been_reached, 
    #     params={
    #         "command_name": "ee_pose",
    #         "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
    #         "ball_cfg": SceneEntityCfg("ball"),
    #         "offset": LINK6_SPOON_OFFSET, # 偏移量要和奖励函数中保持一致
    #         "pos_threshold": 0.05,#0.02
    #         "rot_threshold": 0.5,#0.2
    #     },
    # )
    goal_reached = DoneTerm(
        func=mdp.has_goal_and_ball_been_reached,
        params={
            "command_name": "ee_pose",
            "asset_cfg": SceneEntityCfg("robot", body_names=MISSING),
            "ball_cfg": SceneEntityCfg("ball"),
            "offset": LINK6_SPOON_OFFSET, 
            "pos_threshold": 0.05,
            "rot_threshold": 0.2,
        },
    )

    # ▼▼▼ 球掉了就结束 ▼▼▼
    ball_dropped = DoneTerm(
        # 使用我们新的、返回布尔值的终止函数
        func=mdp.has_ball_dropped,
        params={
        "asset_cfg": SceneEntityCfg("robot", body_names=MISSING), # body_names 在 post_init 中设置
        "ball_cfg": SceneEntityCfg("ball"),
        "offset": LINK6_SPOON_OFFSET,
        "distance_threshold": 0.05,  # 如果球离勺子中心超过5cm，就算掉落
        "grace_period_steps": 120,    # 允许前 120 步内掉球不算数
        },
    )

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class CurriculumCfg:
    """Curriculum terms for the MDP."""
    """为运送任务精心设计的课程学习。"""
  
    #逐渐减小对“慢动作”的强制要求
    #限制机器人动作运动的速度和频率
    #这让机器人从“必须慢”过渡到“可以适当快一些

    action_rate_curriculum = CurrTerm(
        func=mdp.smooth_modify_reward_weight,
        params={"term_name": "action_rate", "weight": -0.1, "num_steps": 3e4}
    )
    joint_vel_curriculum = CurrTerm(
        func=mdp.smooth_modify_reward_weight,
        params={"term_name": "joint_vel", "weight": -0.3, "num_steps": 3e4}
    )
    
    task_reward_curriculum = CurrTerm(
        func=mdp.smooth_modify_reward_weight,
        params={
            "term_name": "task_reward",
            "weight": 400.0,
            "num_steps": 20000
    }
)

    position_tracking_curriculum = CurrTerm(
        func=mdp.smooth_modify_reward_weight, 
        params={"term_name": "end_effector_position_tracking",
                 "weight": -40, 
                 "num_steps": 20000},
    )
    position_tracking_fine_curriculum = CurrTerm(
        func=mdp.smooth_modify_reward_weight, 
        params={"term_name": "end_effector_position_tracking_fine_grained",
                 "weight": 50, 
                 "num_steps": 20000},
    )
    
    orientation_tracking_curriculum = CurrTerm(
        func=mdp.smooth_modify_reward_weight, 
        params={"term_name": "end_effector_orientation_tracking",
                 "weight": -30, # -30
                 "num_steps": 20000},
    )


##
# Environment configuration
##


@configclass
class ReachEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the reach end-effector pose tracking environment."""
    

    # Scene settings
    scene: ReachSceneCfg = ReachSceneCfg(num_envs=4096, env_spacing=2.5)
    # Basic settings
    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    commands: CommandsCfg = CommandsCfg()
    # MDP settings
    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    curriculum: CurriculumCfg = CurriculumCfg()

    def __post_init__(self):
        """Post initialization."""
        # general settings
        self.decimation = 2
        self.sim.render_interval = self.decimation
        self.episode_length_s = 12.0
        self.viewer.eye = (3.5, 3.5, 3.5)
        # simulation settings
        self.sim.dt = 1.0 / 60.0

