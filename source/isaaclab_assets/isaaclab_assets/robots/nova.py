# nova.py
import isaaclab.sim as sim_utils
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.utils.assets import ISAACLAB_NUCLEUS_DIR   # 或你自己的根目录

# -------------------------------------------------------------
# 基础 6-DOF Nova 配置
# -------------------------------------------------------------
NOVA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # ▼ 改成你导出的 Nova 单一 USD 路径
        # usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Nova/nova.usd",
        usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Nova/nova_spoon.usd",
        activate_contact_sensors=False,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=32,
            solver_velocity_iteration_count=1,
        ),
    ),

    init_state=ArticulationCfg.InitialStateCfg(
        # 按照 URDF 中的关节名字填写（六轴）
        joint_pos={
            "joint1": 0.785,    # 45
            "joint2": -0.785,   #-45
            "joint3": -1.571,   #-90
            "joint4": -0.785,   #-45
            "joint5": 0.0,
            "joint6": 1.571,    # 90
            # ▼ 如果你有末端夹爪，再加finger_joint.*": 0.02,
        },
    ),
    
    # init_state=ArticulationCfg.InitialStateCfg(
    #     # 按照 URDF 中的关节名字填写（六轴）
    #     joint_pos={
    #         "joint1": 0.0,   
    #         "joint2": 0.0,   
    #         "joint3": 0.0,   
    #         "joint4": 0.0,   
    #         "joint5": 0.0,
    #         "joint6": 0.0,    
    #         # ▼ 如果你有末端夹爪，再加finger_joint.*": 0.02,
    #     },
    # ),

    actuators={
        # 前 3 轴（腰、肩、肘）
        "nova_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["joint[1-3]"],
            effort_limit_sim=150.0,
            velocity_limit_sim=2.0,
            stiffness=120.0,
            damping=8.0,
        ),
        # 后 3 轴（腕 1/2/3）
        "nova_wrist": ImplicitActuatorCfg(
            joint_names_expr=["joint[4-6]"],
            effort_limit_sim= 50.0,
            velocity_limit_sim=3.0,
            stiffness=100.0,
            damping=5.0,
        ),
        # 夹爪（可选）
        # "nova_gripper": ImplicitActuatorCfg(
        #     joint_names_expr=["nova_finger_joint.*"],
        #     effort_limit_sim=100.0,
        #     velocity_limit_sim=0.1,
        #     stiffness=500.0,
        #     damping=30.0,
        # ),
    },

    soft_joint_pos_limit_factor=1.0,
)

# -------------------------------------------------------------
# 高刚度版本（用于任务空间 IK）
# -------------------------------------------------------------
NOVA_HIGH_PD_CFG = NOVA_CFG.copy()
NOVA_HIGH_PD_CFG.spawn.rigid_props.disable_gravity = True
NOVA_HIGH_PD_CFG.actuators["nova_shoulder"].stiffness = 400.0
NOVA_HIGH_PD_CFG.actuators["nova_shoulder"].damping  = 80.0
NOVA_HIGH_PD_CFG.actuators["nova_wrist"].stiffness   = 400.0
NOVA_HIGH_PD_CFG.actuators["nova_wrist"].damping    = 80.0