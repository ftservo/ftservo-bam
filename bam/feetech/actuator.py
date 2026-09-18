# Copyright 2025 Marc Duclusaud & Grégoire Passault

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

    无内部速率限制（直接跟踪目标位置），使用 ftservo-python-sdk 官方驱动。
    """

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=7.4,
            kp=32,
            # 误差增益：将位置误差 × 固件 KP 转换为 PWM 占空比
            # 标定条件：KP=1 时，4000 计数（≈352°）误差对应 100% PWM
            # 计算：error_gain = 1.0 / (4000 / 651.9) ≈ 0.163
            error_gain=0.163,
            max_pwm=0.97,
        )

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def initialize(self):
        # 扭矩常数 [Nm/A] 或 [V/(rad/s)]
        self.model.kt = Parameter(0.784532, 0.05, 2.5)  # 手册标称 8 kg.cm / A

        self.model.error_gain_ratio = Parameter(1.0, 0.1, 10.0)

        # 电机电阻 [Ohm]
        self.model.R = Parameter(2.0, 0.1, 10.0)

        # 电机转子 / 等效惯量 [kg·m²]
        self.model.armature = Parameter(0.0001, 0.00001, 0.04)

        self.model.q_offset = Parameter(0, -0.2, 0.2)

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        """直接由位置误差计算电压指令（无内部速率限制）。

        :param q_target: 目标关节角 [rad]
        :param q: 当前关节角 [rad]
        :param dq: 当前关节速度 [rad/s]（未使用）
        :param dt: 时间步 [s]（未使用）
        :returns: 输出到电机的电压 [V]
        """
        duty_cycle = (
            (q_target - q)
            * self.kp
            * self.error_gain
            * self.model.error_gain_ratio.value
        )
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty_cycle  # 供日志记录（及电池压降模型）

        return self.vin * duty_cycle


class HD1910Actuator(VoltageControlledActuator):
    """
    Feetech HD1910 7.4v 金属齿轮数字舵机

    与 STS3215 同协议，寄存器地址一致：
    - 位置精度：12 位（4096 计数/圈，0.088°/步）
    - 速度单位：0.732 RPM/单位
    - 电压单位：0.1 V/单位
    - P 增益地址：21
    - 中位位置：2048
    - 死区宽度：≤0.088°

    硬件版本：10.31 (HDS1910)
    固件版本：3.46
    空载速度：150 × 0.732 = 109.8 rpm @7.4V
    """

    def __init__(self, testbench_class: Testbench):
        super().__init__(
            testbench_class,
            vin=7.4,
            kp=32,
            # 误差增益：将位置误差 × 固件 KP 转换为 PWM 占空比
            # 与 STS3215 同协议，相同标定值
            # 标定条件：KP=1 时，4000 计数（≈352°）误差对应 100% PWM
            error_gain=0.163,
            max_pwm=0.97,
        )

    def get_extra_inertia(self) -> float:
        return self.model.armature.value

    def initialize(self):
        # 扭矩常数 [Nm/A]
        # 根据空载速度估算：kt = V / ω = 7.4V / (109.8rpm × 2π/60) ≈ 0.64 Nm/A
        self.model.kt = Parameter(0.64, 0.05, 2.5)

        self.model.error_gain_ratio = Parameter(1.0, 0.1, 10.0)

        # 电机电阻 [Ohm]
        self.model.R = Parameter(5.0, 0.5, 20.0)

        # 电机转子 / 等效惯量 [kg·m²]
        self.model.armature = Parameter(0.00005, 0.000005, 0.02)

        self.model.q_offset = Parameter(0, -0.2, 0.2)

    def compute_control(
        self, q_target: ArrayLike, q: ArrayLike, dq: ArrayLike, dt: float
    ) -> ArrayLike | None:
        """直接由位置误差计算电压指令（无内部速率限制）。

        :param q_target: 目标关节角 [rad]
        :param q: 当前关节角 [rad]
        :param dq: 当前关节速度 [rad/s]（未使用）
        :param dt: 时间步 [s]（未使用）
        :returns: 输出到电机的电压 [V]
        """
        duty_cycle = (
            (q_target - q)
            * self.kp
            * self.error_gain
            * self.model.error_gain_ratio.value
        )
        duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)
        self.duty_cycle = duty_cycle

        return self.vin * duty_cycle
