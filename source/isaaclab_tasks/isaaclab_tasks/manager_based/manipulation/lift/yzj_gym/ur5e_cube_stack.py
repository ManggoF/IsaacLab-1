# Copyright (c) 2021-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import numpy as np
import os

from isaacgym import gymtorch
from isaacgym import gymapi
import torch
from isaacgymenvs.utils.torch_jit_utils import quat_mul, to_torch, tensor_clamp, quat_apply
from isaacgymenvs.tasks.base.vec_task import VecTask


@torch.jit.script
def axisangle2quat(vec, eps=1e-6):
    """
    将缩放的轴角表示转换为四元数
    
    参数:
        vec (Tensor): (..., 3) 张量，最后一维是 (ax,ay,az) 轴角指数坐标
        eps (float): 稳定性阈值，低于该值的小值将映射为0
    
    返回:
        Tensor: (..., 4) 张量，最后一维是 (x,y,z,w) 四元数
    """
    # type: (Tensor, float) -> Tensor
    # 存储输入形状并重塑
    input_shape = vec.shape[:-1]
    vec = vec.reshape(-1, 3)

    # 计算角度
    angle = torch.norm(vec, dim=-1, keepdim=True)

    # 创建返回数组
    quat = torch.zeros(torch.prod(torch.tensor(input_shape)), 4, device=vec.device)
    quat[:, 3] = 1.0

    # 获取角度非零的索引并将输入转换为四元数形式
    idx = angle.reshape(-1) > eps
    quat[idx, :] = torch.cat([
        vec[idx, :] * torch.sin(angle[idx, :] / 2.0) / angle[idx, :],
        torch.cos(angle[idx, :] / 2.0)
    ], dim=-1)

    # 重塑并返回输出
    quat = quat.reshape(list(input_shape) + [4, ])
    return quat


class UR5eCubeStack(VecTask):
    """
    UR5e机械臂堆叠任务环境
    
    任务目标：使用UR5e机械臂将cubeA堆叠到cubeB上方
    """

    def __init__(self, cfg, rl_device, sim_device, graphics_device_id, headless, virtual_screen_capture, force_render):
        """
        初始化UR5e堆叠任务环境
        
        参数:
            cfg (dict): 环境配置字典
            rl_device (str): 强化学习设备
            sim_device (str): 仿真设备
            graphics_device_id (int): 图形设备ID
            headless (bool): 是否无头模式
            virtual_screen_capture (bool): 是否虚拟屏幕捕获
            force_render (bool): 是否强制渲染
        """
        self.cfg = cfg

        self.max_episode_length = self.cfg["env"]["episodeLength"]

        self.action_scale = self.cfg["env"]["actionScale"]
        self.start_position_noise = self.cfg["env"]["startPositionNoise"]
        self.start_rotation_noise = self.cfg["env"]["startRotationNoise"]
        self.ur5e_position_noise = self.cfg["env"]["ur5ePositionNoise"]
        self.ur5e_rotation_noise = self.cfg["env"]["ur5eRotationNoise"]
        self.ur5e_dof_noise = self.cfg["env"]["ur5eDofNoise"]
        self.aggregate_mode = self.cfg["env"]["aggregateMode"]

        # 创建奖励函数参数字典
        self.reward_settings = {
            "r_dist_scale": self.cfg["env"]["distRewardScale"],
            "r_lift_scale": self.cfg["env"]["liftRewardScale"],
            "r_align_scale": self.cfg["env"]["alignRewardScale"],
            "r_stack_scale": self.cfg["env"]["stackRewardScale"],
        }

        # 控制器类型
        self.control_type = self.cfg["env"]["controlType"]
        assert self.control_type in {"osc", "joint_tor"},\
            "无效的控制类型，必须是以下之一: {osc, joint_tor}"

        # UR5e 特定参数
        self.num_arm_dofs = 6  # UR5e有6个臂部自由度
        
        # 计算观测和动作维度（与FrankaCubeStack相同）
        # 基础观测: cubeA_quat (4) + cubeA_pos (3) + cubeA_to_cubeB_pos (3) + eef_pos (3) + eef_quat (4) = 17
        base_obs_dim = 17
        
        if self.control_type == "osc":
            # 添加夹爪关节位置（UR5e有6个夹爪自由度）
            self.cfg["env"]["numObservations"] = base_obs_dim + 6  # 总共23维
        else:  # "joint_tor"
            # 添加所有关节位置（6个臂关节 + 6个夹爪关节 = 12个总自由度）
            self.cfg["env"]["numObservations"] = base_obs_dim + 12  # 总共29维
            
        # 动作空间: 6个臂部动作 + 1个夹爪动作（主关节控制）
        self.cfg["env"]["numActions"] = 7

        # 运行时填充的变量
        self.states = {}                        # 状态字典，用于奖励计算
        self.handles = {}                       # 句柄字典，映射名称到仿真句柄
        self.num_dofs = None                    # 每个环境的总自由度数
        self.actions = None                     # 当前要部署的动作
        self._init_cubeA_state = None           # cubeA的初始状态
        self._init_cubeB_state = None           # cubeB的初始状态
        self._cubeA_state = None                # cubeA的当前状态
        self._cubeB_state = None                # cubeB的当前状态
        self._cubeA_id = None                   # cubeA对应的Actor ID
        self._cubeB_id = None                   # cubeB对应的Actor ID

        # 渐进式探索参数 - 针对主关节 robotiq_85_left_knuckle_joint
        self.initial_angle_limit = 0.1    # 初始角度限制 (约5.7度)
        self.max_angle_limit = 0.5        # 最大角度限制 (约28.6度)
        self.angle_growth_rate = 0.01     # 每次重置增长率
        self.current_angle_limit = self.initial_angle_limit
        
        # 指尖位置偏移量 (URDF局部坐标系x轴方向)
        self.fingertip_offset = 0.04

        # 张量占位符
        self._root_state = None             # 根体状态 (n_envs, 13)
        self._dof_state = None              # 所有关节状态 (n_envs, n_dof)
        self._q = None                      # 关节位置 (n_envs, n_dof)
        self._qd = None                     # 关节速度 (n_envs, n_dof)
        self._rigid_body_state = None       # 所有刚体状态 (n_envs, n_bodies, 13)
        self._contact_forces = None         # 仿真中的接触力
        self._eef_state = None              # 末端执行器状态（在抓取点）
        self._eef_lf_state = None           # 末端执行器状态（在左指尖）
        self._eef_rf_state = None           # 末端执行器状态（在右指尖）
        self._j_eef = None                  # 末端执行器的雅可比矩阵
        self._mm = None                     # 质量矩阵
        self._arm_control = None            # 控制臂的张量缓冲区
        self._gripper_control = None        # 控制夹爪的张量缓冲区
        self._pos_control = None            # 位置动作
        self._effort_control = None         # 力矩动作
        self._ur5e_effort_limits = None     # UR5e的执行器力矩限制
        self._global_indices = None         # 对应展平数组中所有环境的唯一索引

        self.debug_viz = self.cfg["env"]["enableDebugVis"]

        self.up_axis = "z"
        self.up_axis_idx = 2

        # 在调用super().__init__()之前初始化默认DOF位置
        # 稍后在从URDF获知实际DOF数量后会更新
        arm_default_pos = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]  # 中性臂姿态
        self.ur5e_default_dof_pos = None  # 在_create_envs中加载URDF后设置

        super().__init__(config=self.cfg, rl_device=rl_device, sim_device=sim_device, graphics_device_id=graphics_device_id, headless=headless, virtual_screen_capture=virtual_screen_capture, force_render=force_render)

        # OSC增益 - 针对UR5e调优（比Franka更轻的机器人）
        self.kp = to_torch([100.] * 6, device=self.device)  # 位置增益
        self.kd = 2 * torch.sqrt(self.kp)  # 微分增益
        self.kp_null = to_torch([10.] * 6, device=self.device)  # 零空间增益
        self.kd_null = 2 * torch.sqrt(self.kp_null)

        # 控制限制将在_create_envs中加载URDF后设置

        # 重置所有环境
        self.reset_idx(torch.arange(self.num_envs, device=self.device))

        # 刷新张量
        self._refresh()

    def create_sim(self):
        """
        创建仿真环境
        
        无参数
        无返回值
        """
        self.sim_params.up_axis = gymapi.UP_AXIS_Z
        self.sim_params.gravity.x = 0
        self.sim_params.gravity.y = 0
        self.sim_params.gravity.z = -9.81
        self.sim = super().create_sim(
            self.device_id, self.graphics_device_id, self.physics_engine, self.sim_params)
        self._create_ground_plane()
        self._create_envs(self.num_envs, self.cfg["env"]['envSpacing'], int(np.sqrt(self.num_envs)))

    def _create_ground_plane(self):
        """
        创建地面
        
        无参数
        无返回值
        """
        plane_params = gymapi.PlaneParams()
        plane_params.normal = gymapi.Vec3(0.0, 0.0, 1.0)
        self.gym.add_ground(self.sim, plane_params)

    def _create_envs(self, num_envs, spacing, num_per_row):
        """
        创建多个训练环境
        
        参数:
            num_envs (int): 环境数量
            spacing (float): 环境间隔
            num_per_row (int): 每行环境数量
            
        无返回值
        """
        lower = gymapi.Vec3(-spacing, -spacing, 0.0)
        upper = gymapi.Vec3(spacing, spacing, spacing)

        asset_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), "../../assets")
        ur5e_asset_file = self.cfg["env"]["asset"]["assetFileNameUR5e"]

        # 加载UR5e资产
        asset_options = gymapi.AssetOptions()
        asset_options.flip_visual_attachments = True
        asset_options.fix_base_link = True
        asset_options.collapse_fixed_joints = False
        asset_options.disable_gravity = True
        asset_options.thickness = 0.001
        asset_options.default_dof_drive_mode = gymapi.DOF_MODE_EFFORT
        asset_options.use_mesh_materials = True
        ur5e_asset = self.gym.load_asset(self.sim, asset_root, ur5e_asset_file, asset_options)

        # 获取UR5e属性
        self.num_ur5e_bodies = self.gym.get_asset_rigid_body_count(ur5e_asset)
        self.num_ur5e_dofs = self.gym.get_asset_dof_count(ur5e_asset)
        self.num_gripper_dofs = self.num_ur5e_dofs - self.num_arm_dofs

        print("UR5e刚体数量: ", self.num_ur5e_bodies)
        print("UR5e自由度数量: ", self.num_ur5e_dofs)
        print("臂部自由度数量: ", self.num_arm_dofs) 
        print("夹爪自由度数量: ", self.num_gripper_dofs)
        
        # 找到所有夹爪关节的索引，实现手动mimic机制
        self.gripper_joints = {}
        for i in range(self.num_ur5e_dofs):
            dof_name = self.gym.get_asset_dof_name(ur5e_asset, i)
            if dof_name == "robotiq_85_left_knuckle_joint":
                self.gripper_joints['main'] = i - self.num_arm_dofs  # 相对于夹爪的索引
            elif dof_name == "robotiq_85_right_knuckle_joint":
                self.gripper_joints['right_knuckle'] = i - self.num_arm_dofs
            elif dof_name == "robotiq_85_left_inner_knuckle_joint":
                self.gripper_joints['left_inner'] = i - self.num_arm_dofs
            elif dof_name == "robotiq_85_right_inner_knuckle_joint":
                self.gripper_joints['right_inner'] = i - self.num_arm_dofs
            elif dof_name == "robotiq_85_left_finger_tip_joint":
                self.gripper_joints['left_tip'] = i - self.num_arm_dofs
            elif dof_name == "robotiq_85_right_finger_tip_joint":
                self.gripper_joints['right_tip'] = i - self.num_arm_dofs

        # 设置默认位置张量，现在我们知道了实际的DOF数量
        arm_default_pos = [0.0, -1.57, 1.57, -1.57, -1.57, 0.0]
        gripper_default_pos = [0.0] * self.num_gripper_dofs
        self.ur5e_default_dof_pos = to_torch(
            arm_default_pos + gripper_default_pos, device=self.device
        )

        # 设置UR5e DOF属性
        ur5e_dof_stiffness = to_torch([0] * self.num_arm_dofs + [5000.] * self.num_gripper_dofs, dtype=torch.float, device=self.device)
        ur5e_dof_damping = to_torch([0] * self.num_arm_dofs + [1.0e2] * self.num_gripper_dofs, dtype=torch.float, device=self.device)

        # 创建桌子资产
        table_pos = [0.0, 0.0, 1.0]
        table_thickness = 0.05
        table_opts = gymapi.AssetOptions()
        table_opts.fix_base_link = True
        table_asset = self.gym.create_box(self.sim, *[1.2, 1.2, table_thickness], table_opts)

        # 创建桌子支架资产
        table_stand_height = 0.1
        table_stand_pos = [-0.5, 0.0, 1.0 + table_thickness / 2 + table_stand_height / 2]
        table_stand_opts = gymapi.AssetOptions()
        table_stand_opts.fix_base_link = True
        table_stand_asset = self.gym.create_box(self.sim, *[0.2, 0.2, table_stand_height], table_opts)

        self.cubeA_size = 0.050
        self.cubeB_size = 0.070

        # 创建cubeA资产
        cubeA_opts = gymapi.AssetOptions()
        cubeA_asset = self.gym.create_box(self.sim, *([self.cubeA_size] * 3), cubeA_opts)
        cubeA_color = gymapi.Vec3(0.6, 0.1, 0.0)

        # 创建cubeB资产
        cubeB_opts = gymapi.AssetOptions()
        cubeB_asset = self.gym.create_box(self.sim, *([self.cubeB_size] * 3), cubeB_opts)
        cubeB_color = gymapi.Vec3(0.0, 0.4, 0.1)

        # 设置UR5e DOF属性
        ur5e_dof_props = self.gym.get_asset_dof_properties(ur5e_asset)
        self.ur5e_dof_lower_limits = []
        self.ur5e_dof_upper_limits = []
        self._ur5e_effort_limits = []
        for i in range(self.num_ur5e_dofs):
            ur5e_dof_props['driveMode'][i] = gymapi.DOF_MODE_POS if i >= self.num_arm_dofs else gymapi.DOF_MODE_EFFORT
            if self.physics_engine == gymapi.SIM_PHYSX:
                ur5e_dof_props['stiffness'][i] = ur5e_dof_stiffness[i]
                ur5e_dof_props['damping'][i] = ur5e_dof_damping[i]
            else:
                ur5e_dof_props['stiffness'][i] = 7000.0
                ur5e_dof_props['damping'][i] = 50.0

            self.ur5e_dof_lower_limits.append(ur5e_dof_props['lower'][i])
            self.ur5e_dof_upper_limits.append(ur5e_dof_props['upper'][i])
            self._ur5e_effort_limits.append(ur5e_dof_props['effort'][i])

        self.ur5e_dof_lower_limits = to_torch(self.ur5e_dof_lower_limits, device=self.device)
        self.ur5e_dof_upper_limits = to_torch(self.ur5e_dof_upper_limits, device=self.device)
        self._ur5e_effort_limits = to_torch(self._ur5e_effort_limits, device=self.device)
        self.ur5e_dof_speed_scales = torch.ones_like(self.ur5e_dof_lower_limits)
        self.ur5e_dof_speed_scales[self.num_arm_dofs:] = 0.1  # 夹爪移动更慢
        
        # 为夹爪设置更高的力矩限制
        for i in range(self.num_arm_dofs, self.num_ur5e_dofs):
            ur5e_dof_props['effort'][i] = 200

        # 定义UR5e的起始位姿
        ur5e_start_pose = gymapi.Transform()
        ur5e_start_pose.p = gymapi.Vec3(-0.45, 0.0, 1.0 + table_thickness / 2 + table_stand_height)
        ur5e_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # 定义桌子的起始位姿
        table_start_pose = gymapi.Transform()
        table_start_pose.p = gymapi.Vec3(*table_pos)
        table_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        self._table_surface_pos = np.array(table_pos) + np.array([0, 0, table_thickness / 2])
        self.reward_settings["table_height"] = self._table_surface_pos[2]

        # 定义桌子支架的起始位姿
        table_stand_start_pose = gymapi.Transform()
        table_stand_start_pose.p = gymapi.Vec3(*table_stand_pos)
        table_stand_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # 定义方块的起始位姿（不太重要，因为在reset()期间会被覆盖）
        cubeA_start_pose = gymapi.Transform()
        cubeA_start_pose.p = gymapi.Vec3(-1.0, 0.0, 0.0)
        cubeA_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)
        cubeB_start_pose = gymapi.Transform()
        cubeB_start_pose.p = gymapi.Vec3(1.0, 0.0, 0.0)
        cubeB_start_pose.r = gymapi.Quat(0.0, 0.0, 0.0, 1.0)

        # 计算聚合大小
        num_ur5e_bodies = self.gym.get_asset_rigid_body_count(ur5e_asset)
        num_ur5e_shapes = self.gym.get_asset_rigid_shape_count(ur5e_asset)
        max_agg_bodies = num_ur5e_bodies + 4     # 桌子、桌子支架、cubeA、cubeB各1个
        max_agg_shapes = num_ur5e_shapes + 4     # 桌子、桌子支架、cubeA、cubeB各1个

        self.ur5es = []
        self.envs = []

        # 创建环境
        for i in range(self.num_envs):
            # 创建环境实例
            env_ptr = self.gym.create_env(self.sim, lower, upper, num_per_row)

            # 创建actor并根据设置定义适当的聚合组
            # 注意：UR5e应始终在仿真中最先加载！
            if self.aggregate_mode >= 3:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 创建UR5e
            # 可能随机化起始位姿
            if self.ur5e_position_noise > 0:
                rand_xy = self.ur5e_position_noise * (-1. + np.random.rand(2) * 2.0)
                ur5e_start_pose.p = gymapi.Vec3(-0.45 + rand_xy[0], 0.0 + rand_xy[1],
                                                 1.0 + table_thickness / 2 + table_stand_height)
            if self.ur5e_rotation_noise > 0:
                rand_rot = torch.zeros(1, 3)
                rand_rot[:, -1] = self.ur5e_rotation_noise * (-1. + np.random.rand() * 2.0)
                new_quat = axisangle2quat(rand_rot).squeeze().numpy().tolist()
                ur5e_start_pose.r = gymapi.Quat(*new_quat)
            ur5e_actor = self.gym.create_actor(env_ptr, ur5e_asset, ur5e_start_pose, "ur5e", i, 0, 0)
            self.gym.set_actor_dof_properties(env_ptr, ur5e_actor, ur5e_dof_props)

            if self.aggregate_mode == 2:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 创建桌子
            table_actor = self.gym.create_actor(env_ptr, table_asset, table_start_pose, "table", i, 1, 0)
            table_stand_actor = self.gym.create_actor(env_ptr, table_stand_asset, table_stand_start_pose, "table_stand",
                                                      i, 1, 0)

            if self.aggregate_mode == 1:
                self.gym.begin_aggregate(env_ptr, max_agg_bodies, max_agg_shapes, True)

            # 创建方块
            self._cubeA_id = self.gym.create_actor(env_ptr, cubeA_asset, cubeA_start_pose, "cubeA", i, 2, 0)
            self._cubeB_id = self.gym.create_actor(env_ptr, cubeB_asset, cubeB_start_pose, "cubeB", i, 4, 0)
            # 设置颜色
            self.gym.set_rigid_body_color(env_ptr, self._cubeA_id, 0, gymapi.MESH_VISUAL, cubeA_color)
            self.gym.set_rigid_body_color(env_ptr, self._cubeB_id, 0, gymapi.MESH_VISUAL, cubeB_color)

            if self.aggregate_mode > 0:
                self.gym.end_aggregate(env_ptr)

            # 存储创建的环境指针
            self.envs.append(env_ptr)
            self.ur5es.append(ur5e_actor)

        # 设置初始状态缓冲区
        self._init_cubeA_state = torch.zeros(self.num_envs, 13, device=self.device)
        self._init_cubeB_state = torch.zeros(self.num_envs, 13, device=self.device)

        # 现在我们有了力矩限制，设置控制限制
        self.cmd_limit = to_torch([0.1, 0.1, 0.1, 0.5, 0.5, 0.5], device=self.device).unsqueeze(0) if \
        self.control_type == "osc" else self._ur5e_effort_limits[:6].unsqueeze(0)

        # 设置数据
        self.init_data()

    def init_data(self):
        """
        初始化数据结构和张量
        
        无参数
        无返回值
        """
        # 设置仿真句柄
        env_ptr = self.envs[0]
        ur5e_handle = 0
        self.handles = {
            # UR5e - 使用URDF中正确的链接名称
            "hand": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle, "robotiq_85_base_link"),
            "leftfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle, "robotiq_85_left_finger_tip_link"),
            "rightfinger_tip": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle, "robotiq_85_right_finger_tip_link"),
            "grip_site": self.gym.find_actor_rigid_body_handle(env_ptr, ur5e_handle, "ur5e_grip_site"),
            # 方块
            "cubeA_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeA_id, "box"),
            "cubeB_body_handle": self.gym.find_actor_rigid_body_handle(self.envs[0], self._cubeB_id, "box"),
        }

        # 获取总自由度数
        self.num_dofs = self.gym.get_sim_dof_count(self.sim) // self.num_envs

        # 设置张量缓冲区
        _actor_root_state_tensor = self.gym.acquire_actor_root_state_tensor(self.sim)
        _dof_state_tensor = self.gym.acquire_dof_state_tensor(self.sim)
        _rigid_body_state_tensor = self.gym.acquire_rigid_body_state_tensor(self.sim)
        self._root_state = gymtorch.wrap_tensor(_actor_root_state_tensor).view(self.num_envs, -1, 13)
        self._dof_state = gymtorch.wrap_tensor(_dof_state_tensor).view(self.num_envs, -1, 2)
        self._rigid_body_state = gymtorch.wrap_tensor(_rigid_body_state_tensor).view(self.num_envs, -1, 13)
        self._q = self._dof_state[..., 0]
        self._qd = self._dof_state[..., 1]
        self._eef_state = self._rigid_body_state[:, self.handles["grip_site"], :]
        self._eef_lf_state = self._rigid_body_state[:, self.handles["leftfinger_tip"], :]
        self._eef_rf_state = self._rigid_body_state[:, self.handles["rightfinger_tip"], :]
        
        # 设置雅可比矩阵和质量矩阵
        _jacobian = self.gym.acquire_jacobian_tensor(self.sim, "ur5e")
        jacobian = gymtorch.wrap_tensor(_jacobian)
        hand_joint_index = self.gym.get_actor_joint_dict(env_ptr, ur5e_handle)['ee_fixed_joint']
        self._j_eef = jacobian[:, hand_joint_index, :, :self.num_arm_dofs]
        _massmatrix = self.gym.acquire_mass_matrix_tensor(self.sim, "ur5e")
        mm = gymtorch.wrap_tensor(_massmatrix)
        self._mm = mm[:, :self.num_arm_dofs, :self.num_arm_dofs]
        
        self._cubeA_state = self._root_state[:, self._cubeA_id, :]
        self._cubeB_state = self._root_state[:, self._cubeB_id, :]

        # 初始化状态
        self.states.update({
            "cubeA_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeA_size,
            "cubeB_size": torch.ones_like(self._eef_state[:, 0]) * self.cubeB_size,
        })

        # 初始化动作
        self._pos_control = torch.zeros((self.num_envs, self.num_dofs), dtype=torch.float, device=self.device)
        self._effort_control = torch.zeros_like(self._pos_control)

        # 初始化控制
        self._arm_control = self._effort_control[:, :self.num_arm_dofs]
        self._gripper_control = self._pos_control[:, self.num_arm_dofs:]

        # 初始化索引
        self._global_indices = torch.arange(self.num_envs * 5, dtype=torch.int32,
                                           device=self.device).view(self.num_envs, -1)

    def _update_states(self):
        """
        更新环境状态
        
        无参数
        无返回值
        """
        # 计算真实指尖位置 (应用0.04m偏移)
        lf_quat = self._eef_lf_state[:, 3:7]
        rf_quat = self._eef_rf_state[:, 3:7]
        
        # 创建指尖偏移向量 (局部坐标系x轴方向)
        fingertip_offset_vec = to_torch([self.fingertip_offset, 0, 0], device=self.device).repeat(self.num_envs, 1)
        
        # 应用旋转变换得到世界坐标系下的实际指尖位置
        eef_lf_pos_actual = self._eef_lf_state[:, :3] + quat_apply(lf_quat, fingertip_offset_vec)
        eef_rf_pos_actual = self._eef_rf_state[:, :3] + quat_apply(rf_quat, fingertip_offset_vec)
        
        # 计算抓取中心位置 (两个实际指尖位置的中点)
        eef_pos_actual = (eef_lf_pos_actual + eef_rf_pos_actual) / 2.0
        
        self.states.update({
            # UR5e
            "q": self._q[:, :],
            "q_gripper": self._q[:, self.num_arm_dofs:],
            "eef_pos": eef_pos_actual,  # 使用计算的抓取中心
            "eef_quat": self._eef_state[:, 3:7],  # 保持原始grip_site的姿态
            "eef_vel": self._eef_state[:, 7:],
            "eef_lf_pos": eef_lf_pos_actual,  # 真实左指尖位置
            "eef_rf_pos": eef_rf_pos_actual,  # 真实右指尖位置
            # 方块
            "cubeA_quat": self._cubeA_state[:, 3:7],
            "cubeA_pos": self._cubeA_state[:, :3],
            "cubeA_pos_relative": self._cubeA_state[:, :3] - eef_pos_actual,  # 使用新的抓取中心
            "cubeB_quat": self._cubeB_state[:, 3:7],
            "cubeB_pos": self._cubeB_state[:, :3],
            "cubeA_to_cubeB_pos": self._cubeB_state[:, :3] - self._cubeA_state[:, :3],
        })

    def _refresh(self):
        """
        刷新仿真状态张量
        
        """
        self.gym.refresh_actor_root_state_tensor(self.sim)
        self.gym.refresh_dof_state_tensor(self.sim)
        self.gym.refresh_rigid_body_state_tensor(self.sim)
        self.gym.refresh_jacobian_tensors(self.sim)
        self.gym.refresh_mass_matrix_tensors(self.sim)

        # 刷新状态
        self._update_states()

    def compute_reward(self, actions):
        """
        计算奖励
        
        参数:
            actions (Tensor): 动作张量
            
        """
        self.rew_buf[:], self.reset_buf[:] = compute_ur5e_reward(
            self.reset_buf, self.progress_buf, self.actions, self.states, self.reward_settings, 
            self.max_episode_length
        )

    def compute_observations(self):
        """
        计算观测
        
        无参数
        
        返回:
            Tensor: 观测张量
        """
        self._refresh()
        
        # 简单的观测空间（与FrankaCubeStack相同）
        obs = ["cubeA_quat", "cubeA_pos", "cubeA_to_cubeB_pos", "eef_pos", "eef_quat"]
        obs += ["q_gripper"] if self.control_type == "osc" else ["q"]
        
        self.obs_buf = torch.cat([self.states[ob] for ob in obs], dim=-1)

        return self.obs_buf

    def reset_idx(self, env_ids):
        """
        重置指定环境
        
        参数:
            env_ids (Tensor): 需要重置的环境ID
            
        """
        env_ids_int32 = env_ids.to(dtype=torch.int32)

        # 重置方块，先采样方块B，然后采样方块A
        self._reset_init_cube_state(cube='B', env_ids=env_ids, check_valid=False)
        self._reset_init_cube_state(cube='A', env_ids=env_ids, check_valid=True)

        # 将这些新的初始状态写入仿真状态
        self._cubeA_state[env_ids] = self._init_cubeA_state[env_ids]
        self._cubeB_state[env_ids] = self._init_cubeB_state[env_ids]

        # 重置机器人
        reset_noise = torch.rand((len(env_ids), self.num_ur5e_dofs), device=self.device)
        pos = tensor_clamp(
            self.ur5e_default_dof_pos.unsqueeze(0) +
            self.ur5e_dof_noise * 2.0 * (reset_noise - 0.5),
            self.ur5e_dof_lower_limits.unsqueeze(0), self.ur5e_dof_upper_limits.unsqueeze(0))

        # 覆盖夹爪初始位置（无噪声，因为这些总是位置控制）
        pos[:, self.num_arm_dofs:] = self.ur5e_default_dof_pos[self.num_arm_dofs:]

        # 相应地重置内部观测
        self._q[env_ids, :] = pos
        self._qd[env_ids, :] = torch.zeros_like(self._qd[env_ids])

        # 将任何位置控制设置为当前位置，并将任何速度/力矩控制设置为0
        # 注意：任务使用SimActions API在仿真中实际传播这些控制
        self._pos_control[env_ids, :] = pos
        self._effort_control[env_ids, :] = torch.zeros_like(pos)

        # 部署更新
        multi_env_ids_int32 = self._global_indices[env_ids, 0].flatten()
        self.gym.set_dof_position_target_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._pos_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_actuation_force_tensor_indexed(self.sim,
                                                        gymtorch.unwrap_tensor(self._effort_control),
                                                        gymtorch.unwrap_tensor(multi_env_ids_int32),
                                                        len(multi_env_ids_int32))
        self.gym.set_dof_state_tensor_indexed(self.sim,
                                              gymtorch.unwrap_tensor(self._dof_state),
                                              gymtorch.unwrap_tensor(multi_env_ids_int32),
                                              len(multi_env_ids_int32))

        # 更新方块状态
        multi_env_ids_cubes_int32 = self._global_indices[env_ids, -2:].flatten()
        self.gym.set_actor_root_state_tensor_indexed(
            self.sim, gymtorch.unwrap_tensor(self._root_state),
            gymtorch.unwrap_tensor(multi_env_ids_cubes_int32), len(multi_env_ids_cubes_int32))

        self.progress_buf[env_ids] = 0
        self.reset_buf[env_ids] = 0
        
        # 更新渐进式探索参数 (每次重置时增加角度限制)
        if len(env_ids) > 0:
            self.current_angle_limit = min(
                self.current_angle_limit + self.angle_growth_rate,
                self.max_angle_limit
            )

    def _reset_init_cube_state(self, cube, env_ids, check_valid=True):
        """
        采样方块位置并重置其姿态
        
        基于startPositionNoise和startRotationNoise采样方块的位置，
        并自动在内部重置姿态。填充相应的self._init_cubeX_state。
        
        如果check_valid为True，则还会确保采样的位置与另一个方块没有接触。
        
        参数:
            cube (str): 要采样位置的方块。'A' 或 'B'
            env_ids (Tensor or None): 要为其重置方块的特定环境
            check_valid (bool): 是否确保采样位置与另一个方块无碰撞
    
        """
        # 如果env_ids为None，我们重置所有环境
        if env_ids is None:
            env_ids = torch.arange(start=0, end=self.num_envs, device=self.device, dtype=torch.long)

        # 初始化缓冲区以保存采样值
        num_resets = len(env_ids)
        sampled_cube_state = torch.zeros(num_resets, 13, device=self.device)

        # 根据选择的方块获取正确的引用
        if cube.lower() == 'a':
            this_cube_state_all = self._init_cubeA_state
            other_cube_state = self._init_cubeB_state[env_ids, :]
            cube_heights = self.states["cubeA_size"]
        elif cube.lower() == 'b':
            this_cube_state_all = self._init_cubeB_state
            other_cube_state = self._init_cubeA_state[env_ids, :]
            cube_heights = self.states["cubeB_size"]
        else:
            raise ValueError(f"无效的方块指定，选项为'A'和'B'；得到: {cube}")

        # 保证无碰撞采样的最小方块距离是每个方块有效半径之和
        min_dists = (self.states["cubeA_size"] + self.states["cubeB_size"])[env_ids] * np.sqrt(2) / 2.0

        # 将最小距离缩放2倍，使方块不会太近
        min_dists = min_dists * 2.0

        centered_cube_xy_state = torch.tensor(self._table_surface_pos[:2], device=self.device, dtype=torch.float32)

        # 设置z值，即固定高度
        sampled_cube_state[:, 2] = self._table_surface_pos[2] + cube_heights.squeeze(-1)[env_ids] / 2

        # 初始化旋转，不旋转（四元数w = 1）
        sampled_cube_state[:, 6] = 1.0

        # 使用一个简单的启发式方法，基于方块的半径来检查是否会发生碰撞
        if check_valid:
            success = False
            # 对应于仍在积极采样的环境的索引
            active_idx = torch.arange(num_resets, device=self.device)
            num_active_idx = len(active_idx)
            for i in range(100):
                # 采样x y值
                sampled_cube_state[active_idx, :2] = centered_cube_xy_state + \
                                                     2.0 * self.start_position_noise * (
                                                             torch.rand_like(sampled_cube_state[active_idx, :2]) - 0.5)
                # 检查采样值是否有效
                cube_dist = torch.linalg.norm(sampled_cube_state[:, :2] - other_cube_state[:, :2], dim=-1)
                active_idx = torch.nonzero(cube_dist < min_dists, as_tuple=True)[0]
                num_active_idx = len(active_idx)
                # 如果活跃索引为空，则所有采样都是有效的 :D
                if num_active_idx == 0:
                    success = True
                    break
            # 确保成功采样
            assert success, "采样方块位置失败！ ):"
        else:
            sampled_cube_state[:, :2] = centered_cube_xy_state.unsqueeze(0) + \
                                              2.0 * self.start_position_noise * (
                                                      torch.rand(num_resets, 2, device=self.device) - 0.5)

        # 采样旋转值
        if self.start_rotation_noise > 0:
            aa_rot = torch.zeros(num_resets, 3, device=self.device)
            aa_rot[:, 2] = 2.0 * self.start_rotation_noise * (torch.rand(num_resets, device=self.device) - 0.5)
            sampled_cube_state[:, 3:7] = quat_mul(axisangle2quat(aa_rot), sampled_cube_state[:, 3:7])

        # 将采样值设置为新的初始状态
        this_cube_state_all[env_ids, :] = sampled_cube_state

    def _compute_osc_torques(self, dpose):
        """
        计算操作空间控制（OSC）力矩
        
        参数:
            dpose (Tensor): 笛卡尔空间位姿变化 (batch_size, 6)
            
        返回:
            Tensor: 关节力矩 (batch_size, num_arm_dofs)
        """
        q, qd = self._q[:, :self.num_arm_dofs], self._qd[:, :self.num_arm_dofs]
        mm_inv = torch.inverse(self._mm)
        m_eef_inv = self._j_eef @ mm_inv @ torch.transpose(self._j_eef, 1, 2)
        m_eef = torch.inverse(m_eef_inv)

        # 将笛卡尔动作dpose转换为关节力矩u
        u = torch.transpose(self._j_eef, 1, 2) @ m_eef @ (
                self.kp * dpose - self.kd * self.states["eef_vel"][:, :6]).unsqueeze(-1)

        # 零空间控制力矩u_null防止关节配置的大幅变化
        # 它们被添加到OSC的零空间中，以便末端执行器方向保持不变

        j_eef_inv = m_eef @ self._j_eef @ mm_inv
        u_null = self.kd_null * -qd + self.kp_null * (
                (self.ur5e_default_dof_pos[:self.num_arm_dofs] - q + np.pi) % (2 * np.pi) - np.pi)
        u_null = self._mm @ u_null.unsqueeze(-1)
        u += (torch.eye(self.num_arm_dofs, device=self.device).unsqueeze(0) - torch.transpose(self._j_eef, 1, 2) @ j_eef_inv) @ u_null

        # 将值限制在有效力矩范围内
        u = tensor_clamp(u.squeeze(-1),
                         -self._ur5e_effort_limits[:self.num_arm_dofs].unsqueeze(0), self._ur5e_effort_limits[:self.num_arm_dofs].unsqueeze(0))

        return u

    def _compute_progressive_gripper_control(self, u_gripper):
        """
        渐进式主关节控制 - 实现完整的mimic机制
        
        特点:
        1. 主关节控制: 只学习robotiq_85_left_knuckle_joint，其他关节通过手动mimic跟随
        2. 渐进式探索: 从小角度开始，逐步扩大探索范围 
        3. 相对控制: 基于当前位置的增量控制，避免位置突变
        
        Args:
            u_gripper: 夹爪动作 (batch_size,) - 主关节的增量动作
                      动作范围 [-1, 1]，表示角度增量的比例
        Returns:
            u_fingers: 夹爪关节位置指令 (batch_size, num_gripper_dofs)
        """
        u_fingers = torch.zeros_like(self._gripper_control)
        
        # 获取当前所有夹爪关节位置
        current_gripper_pos = self._q[:, self.num_arm_dofs:]
        
        if 'main' in self.gripper_joints:
            main_idx = self.gripper_joints['main']
            
            # 获取主关节当前位置
            main_joint_current_pos = current_gripper_pos[:, main_idx]
            
            # 计算角度增量 (基于当前探索限制)
            angle_increment = u_gripper * self.current_angle_limit
            
            # 计算目标位置
            target_pos = main_joint_current_pos + angle_increment
            
            # 应用主关节限制
            main_joint_lower = self.ur5e_dof_lower_limits[self.num_arm_dofs + main_idx]
            main_joint_upper = self.ur5e_dof_upper_limits[self.num_arm_dofs + main_idx]
            target_pos_clamped = torch.clamp(target_pos, main_joint_lower, main_joint_upper)
            
            # 设置主关节
            u_fingers[:, main_idx] = target_pos_clamped
            
            # 设置mimic关节 (跟随主关节)
            if 'right_knuckle' in self.gripper_joints:
                u_fingers[:, self.gripper_joints['right_knuckle']] = target_pos_clamped
            if 'left_inner' in self.gripper_joints:
                u_fingers[:, self.gripper_joints['left_inner']] = target_pos_clamped
            if 'right_inner' in self.gripper_joints:
                u_fingers[:, self.gripper_joints['right_inner']] = target_pos_clamped
                
            # 设置mimic关节 (multiplier=-1，反向跟随)
            if 'left_tip' in self.gripper_joints:
                u_fingers[:, self.gripper_joints['left_tip']] = -target_pos_clamped
            if 'right_tip' in self.gripper_joints:
                u_fingers[:, self.gripper_joints['right_tip']] = -target_pos_clamped
        else:
            # 备用方案：如果没找到主关节，保持当前位置
            u_fingers[:, :] = current_gripper_pos
        
        return u_fingers

    def pre_physics_step(self, actions):
        """
        物理步骤之前的预处理
        
        参数:
            actions (Tensor): 动作张量 (batch_size, num_actions)
            
        """
        self.actions = actions.clone().to(self.device)

        # 拆分臂和夹爪命令
        u_arm, u_gripper = self.actions[:, :self.num_arm_dofs], self.actions[:, self.num_arm_dofs]

        # 控制臂（首先缩放值）
        u_arm = u_arm * self.cmd_limit / self.action_scale
        if self.control_type == "osc":
            u_arm = self._compute_osc_torques(dpose=u_arm)
        self._arm_control[:, :] = u_arm

        # 渐进式夹爪控制，仅使用主关节
        u_fingers = self._compute_progressive_gripper_control(u_gripper)
        
        # 将夹爪命令写入适当的张量缓冲区
        self._gripper_control[:, :] = u_fingers

        # 部署动作
        self.gym.set_dof_position_target_tensor(self.sim, gymtorch.unwrap_tensor(self._pos_control))
        self.gym.set_dof_actuation_force_tensor(self.sim, gymtorch.unwrap_tensor(self._effort_control))

    def post_physics_step(self):
        """
        物理步骤之后的后处理
        """
        self.progress_buf += 1

        env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1)
        if len(env_ids) > 0:
            self.reset_idx(env_ids)

        self.compute_observations()
        self.compute_reward(self.actions)

        # 调试可视化
        if self.viewer and self.debug_viz:
            self.gym.clear_lines(self.viewer)
            self.gym.refresh_rigid_body_state_tensor(self.sim)

            # 抓取相关状态以可视化
            eef_pos = self.states["eef_pos"]
            eef_rot = self.states["eef_quat"]
            cubeA_pos = self.states["cubeA_pos"]
            cubeA_rot = self.states["cubeA_quat"]
            cubeB_pos = self.states["cubeB_pos"]
            cubeB_rot = self.states["cubeB_quat"]

            # 绘制可视化
            for i in range(self.num_envs):
                for pos, rot in zip((eef_pos, cubeA_pos, cubeB_pos), (eef_rot, cubeA_rot, cubeB_rot)):
                    px = (pos[i] + quat_apply(rot[i], to_torch([1, 0, 0], device=self.device) * 0.2)).cpu().numpy()
                    py = (pos[i] + quat_apply(rot[i], to_torch([0, 1, 0], device=self.device) * 0.2)).cpu().numpy()
                    pz = (pos[i] + quat_apply(rot[i], to_torch([0, 0, 1], device=self.device) * 0.2)).cpu().numpy()

                    p0 = pos[i].cpu().numpy()
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], px[0], px[1], px[2]], [0.85, 0.1, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], py[0], py[1], py[2]], [0.1, 0.85, 0.1])
                    self.gym.add_lines(self.viewer, self.envs[i], 1, [p0[0], p0[1], p0[2], pz[0], pz[1], pz[2]], [0.1, 0.1, 0.85])

#####################################################################
###=========================JIT函数=========================###
#####################################################################


@torch.jit.script 
def compute_ur5e_reward(
    reset_buf, progress_buf, actions, states, reward_settings, max_episode_length
):
    """
    计算UR5e堆叠任务的奖励
    
    参数:
        reset_buf (Tensor): 重置缓冲区
        progress_buf (Tensor): 进度缓冲区
        actions (Tensor): 动作张量
        states (Dict[str, Tensor]): 状态字典
        reward_settings (Dict[str, float]): 奖励设置字典
        max_episode_length (float): 最大回合长度
        
    返回:
        Tuple[Tensor, Tensor]: (奖励, 重置标志)
    """
    # type: (Tensor, Tensor, Tensor, Dict[str, Tensor], Dict[str, float], float) -> Tuple[Tensor, Tensor]

    # 计算每个环境的物理参数
    target_height = states["cubeB_size"] + states["cubeA_size"] / 2.0
    cubeA_size = states["cubeA_size"]
    cubeB_size = states["cubeB_size"]

    # 从手到cubeA的距离
    d = torch.norm(states["cubeA_pos_relative"], dim=-1)
    d_lf = torch.norm(states["cubeA_pos"] - states["eef_lf_pos"], dim=-1)
    d_rf = torch.norm(states["cubeA_pos"] - states["eef_rf_pos"], dim=-1)
    dist_reward = 1 - torch.tanh(10.0 * (d + d_lf + d_rf) / 3)

    # 提升cubeA的奖励
    cubeA_height = states["cubeA_pos"][:, 2] - reward_settings["table_height"]
    cubeA_lifted = (cubeA_height - cubeA_size.squeeze(-1)) > 0.04
    lift_reward = cubeA_lifted

    # cubeA与cubeB的对齐程度
    offset = torch.zeros_like(states["cubeA_to_cubeB_pos"])
    offset[:, 2] = (cubeA_size + cubeB_size) / 2
    d_ab = torch.norm(states["cubeA_to_cubeB_pos"] + offset, dim=-1)
    align_reward = (1 - torch.tanh(10.0 * d_ab)) * cubeA_lifted

    # 距离奖励是dist和align奖励的最大值
    dist_reward = torch.max(dist_reward, align_reward)

    # 成功堆叠的最终奖励
    cubeA_align_cubeB = (torch.norm(states["cubeA_to_cubeB_pos"][:, :2], dim=-1) < 0.02)
    cubeA_on_cubeB = torch.abs(cubeA_height - target_height.squeeze(-1)) < 0.02
    gripper_away_from_cubeA = (d > 0.04)
    stack_reward = cubeA_align_cubeB & cubeA_on_cubeB & gripper_away_from_cubeA

    rewards = torch.where(
        stack_reward,
        reward_settings["r_stack_scale"] * stack_reward,
        reward_settings["r_dist_scale"] * dist_reward + reward_settings["r_lift_scale"] * lift_reward + reward_settings["r_align_scale"] * align_reward,
    )

    # 计算重置
    reset_buf = torch.where((progress_buf >= max_episode_length - 1) | (stack_reward > 0), torch.ones_like(reset_buf), reset_buf)

    return rewards, reset_buf
