# custom_actions.py

from __future__ import annotations

import torch
from dataclasses import MISSING

from isaaclab.envs.mdp.actions import JointPositionAction, JointPositionActionCfg
from isaaclab.utils import configclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


@configclass
class SymmetricJointPositionActionCfg(JointPositionActionCfg):
    """
    一个特殊的关节位置动作配置，它只从策略网络接收一个动作值，
    然后根据指定的方向将其应用到两个关节上，以实现对称运动。
    """
    num_actions: int = 1
    direction: dict[str, float] = MISSING


class SymmetricJointPositionAction(JointPositionAction):
    """SymmetricJointPositionAction的实现类。"""
    cfg: SymmetricJointPositionActionCfg
    _direction_multipliers: torch.Tensor

    def __init__(self, cfg: SymmetricJointPositionActionCfg, env: ManagerBasedRLEnv):
        # 父类的 __init__ 会正确地设置好 self._asset, self._joint_ids,
        # self._scale, self._offset, self._raw_actions 等所有必需的变量。
        super().__init__(cfg, env)

        # 在初始化时就创建好方向乘数张量。
        direction_values = [self.cfg.direction[j] for j in self.cfg.joint_names]
        self._direction_multipliers = torch.tensor(direction_values, device=self.device)

    def apply_actions(self):
        """
        这个方法由 ActionManager 在每个模拟步骤调用。
        它负责完成从原始动作到最终物理指令的全部流程。
        """
        # 1. [自定义逻辑]: 将 self._raw_actions (由Manager更新) 对称化
        symmetric_actions = self._raw_actions * self._direction_multipliers

        # 2. [父类逻辑]: 执行缩放和偏移
        processed_actions = self._offset + self._scale * symmetric_actions

        # 3. [父类逻辑]: 执行裁剪
        if self.cfg.clip is not None:
            processed_actions = torch.clamp(processed_actions, self._clip[..., 0], self._clip[..., 1])

        # 4. [父类逻辑]: 将最终计算出的指令应用到机器人上
        self._asset.set_joint_position_target(processed_actions, joint_ids=self._joint_ids)