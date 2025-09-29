# # Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# # All rights reserved.
# #
# # SPDX-License-Identifier: BSD-3-Clause

# from __future__ import annotations
# import torch
# from typing import TYPE_CHECKING

# from isaaclab.utils import configclass
# import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

# if TYPE_CHECKING:
#     from isaaclab.envs import ManagerBasedRLEnv




# class DelayedJointPositionAction(mdp.JointPositionAction):
#     """
#     一个带有初始延迟的关节位置动作实现。
#     """
#     cfg: DelayedJointPositionActionCfg

#     def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRLEnv):
#         super().__init__(cfg, env)
#         self._dt = env.step_dt


#     def __call__(self, env: ManagerBasedRLEnv, actions: torch.Tensor):
#         """
#         处理输入的动作。如果在延迟期内，则用零覆盖动作。
#         """
#         print("★★★ DELAYED ACTION call WAS EXECUTED! ★★★")
#         # 计算每个环境的当前时间
#         current_time_s = (env.episode_length_buf * self._dt)

#         # 创建一个掩码，标记哪些环境仍在延迟期内
#         # unsqueeze(1) 是为了让它可以广播到 (num_envs, num_actions) 的形状
#         is_delayed = (current_time_s < self.cfg.delay_s).unsqueeze(1)

#         # 如果在延迟期内，使用零动作；否则，使用策略网络提供的动作
#         processed_actions = torch.where(is_delayed, 0.0, actions)

#         # 调用父类的 __call__ 方法来应用缩放、偏移等
#         super().__call__(env, processed_actions)


# @configclass
# class DelayedJointPositionActionCfg(mdp.JointPositionActionCfg):
#     """
#     一个带有初始延迟的关节位置动作配置。
#     在前 `delay_s` 秒内，此动作项将忽略策略的输出，强制发送零动作。
#     """
#     # ▼▼▼ 关键修复: 添加这行代码 ▼▼▼
#     # 这行代码告诉框架，当使用这个配置时，应该创建 DelayedJointPositionAction 类的实例。
#     class_type: type = DelayedJointPositionAction
#     # 延迟时间（秒），在此期间机器人将保持静止
#     delay_s: float = 0.5

# Copyright (c) 2022-2025, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations
import torch
from typing import TYPE_CHECKING

from isaaclab.utils import configclass
# 导入您环境中的 mdp 模块，以获取父类 JointPositionAction
import isaaclab_tasks.manager_based.manipulation.reach.mdp as mdp

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


class DelayedJointPositionAction(mdp.JointPositionAction):
    """
    一个带有初始延迟的关节位置动作实现。
    它通过重写 process_actions 方法来注入延迟逻辑。
    """
    cfg: DelayedJointPositionActionCfg

    def __init__(self, cfg: DelayedJointPositionActionCfg, env: ManagerBasedRLEnv):
        # 调用父类的初始化，这很重要
        super().__init__(cfg, env)
        # 获取环境的时间步长，用于计算当前时间
        self._dt = env.step_dt

    def process_actions(self, actions: torch.Tensor):
        """
        处理输入的动作。这是框架在每个环境步骤中真正调用的方法。
        """

        # 1. 计算每个环境的当前运行时间
        current_time_s = self._env.episode_length_buf * self._dt

        # 2. 创建一个掩码，标记哪些环境仍在延迟期内
        #    unsqueeze(1) 是为了让它可以广播到 (num_envs, num_actions) 的形状
        is_delayed = (current_time_s < self.cfg.delay_s).unsqueeze(1)

        # 3. 如果在延迟期内，使用零动作；否则，使用策略网络提供的原始动作
        #    零动作经过父类的处理（乘以scale，加上offset）后，就会变成“保持初始姿态”的指令
        processed_actions = torch.where(is_delayed, 0.0, actions)

        # 4. 调用父类的 process_actions 方法
        #    这至关重要，因为它会负责应用 scale 和 offset，并把最终结果存入 self._processed_actions
        super().process_actions(processed_actions)


@configclass
class DelayedJointPositionActionCfg(mdp.JointPositionActionCfg):
    """
    一个带有初始延迟的关节位置动作配置。
    在前 `delay_s` 秒内，此动作项将忽略策略的输出，强制发送零动作。
    """
    # 关键链接：告诉框架，当使用这个配置时，应该创建我们下面的 DelayedJointPositionAction 类的实例。
    class_type: type = DelayedJointPositionAction  # Forward declaration
    # 延迟时间（秒），在此期间机器人将保持静止
    delay_s: float = 0.25
