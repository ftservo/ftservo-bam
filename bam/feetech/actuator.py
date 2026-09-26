# Copyright 2026 ftservo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from __future__ import annotations

from typing import TYPE_CHECKING
from bam.actuator import VoltageControlledActuator
from bam.parameter import Parameter
from bam.testbench import Testbench

if TYPE_CHECKING:
    from bam.actuator import ArrayLike

class STS3215Actuator(VoltageControlledActuator):
    """
    Feetech STS3215 7.4v
    """

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=7.4,
            kp=32,
            error_gain=0.163,
            max_pwm=0.97,
        )
        self.q_target_smooth = None  # 内部平滑目标位置（有状态）

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        # 第一次调用，或者 reset 后：内部目标 = 当前位置
        if self.q_target_smooth is None:
            self.q_target_smooth = q

        # 速率限制：内部目标最多以 max_velocity 的速度靠近目标
        max_step = self.model.max_velocity.value * dt
        self.q_target_smooth = self.backend.clamp(
            q_target,
            self.q_target_smooth - max_step,
            self.q_target_smooth + max_step,
        )

        # P 控制器：误差 = 平滑后的内部目标 - 当前位置
        duty_cycle = (
            (self.q_target_smooth - q)
            * self.kp
            * self.error_gain
            * self.model.error_gain_ratio.value
        )
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty_cycle

        return self.vin * duty_cycle

    def initialize(self):
        self.model.kt = Parameter(0.784532, 0.05, 2.5)
        self.model.error_gain_ratio = Parameter(1.0, 0.1, 10.0)
        self.model.R = Parameter(2.0, 0.1, 10.0)
        self.model.armature = Parameter(0.024, 0.001, 0.08)
        self.model.max_velocity = Parameter(7.0, 1.0, 15.0)
        self.model.q_offset = Parameter(0, -0.2, 0.2)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

class HD1910Actuator(VoltageControlledActuator):
    """
    Feetech HD1910 7.4v 金属齿轮数字舵机
    """

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=7.4,
            kp=32,
            error_gain=0.163,
            max_pwm=0.97,
        )

    def initialize(self):
        self.model.kt = Parameter(0.61, 0.3, 1.0)
        self.model.R = Parameter(5, 2.0, 7.0)
        self.model.armature = Parameter(0.001, 0.0001, 0.007)

    def get_extra_inertia(self) -> float:
        return self.model.armature.value
