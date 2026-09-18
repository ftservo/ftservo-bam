# Copyright 2025 Marc Duclusaud & Grégoire Passault

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

from scservo_sdk import PortHandler, COMM_SUCCESS
from scservo_sdk.sms_sts import sms_sts, SMS_STS_TORQUE_ENABLE, SMS_STS_GOAL_POSITION_L
from scservo_sdk.sms_sts import SMS_STS_PRESENT_POSITION_L, SMS_STS_PRESENT_SPEED_L
from scservo_sdk.sms_sts import SMS_STS_PRESENT_VOLTAGE, SMS_STS_PRESENT_TEMPERATURE
import json
import datetime
import os
import numpy as np
import argparse
import time
from bam.trajectory import *

# P 增益寄存器地址（不同舵机型号地址不同）
# STS3215: 地址 21 (0x15)
# HD1910:  地址 50 (0x32)
ADDR_P_GAIN_MAP = {
    "sts3215": 21,
    "hd1910": 50,
}

# 位置换算：12 位有效精度，4096 计数 = 2π（一圈）
# STS3215 和 HD1910 通用
POS_SCALE = 4096.0 / (2 * np.pi)  # rad → 原始值

# 速度单位换算：原始值 × 0.732 RPM → rad/s
# STS3215 和 HD1910 通用
# 1 RPM = 2π/60 rad/s
RPM_TO_RAD_S = 2 * np.pi / 60
SPEED_SCALE = 0.732 * RPM_TO_RAD_S  # 原始值 → rad/s

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--mass", type=float, required=True)
arg_parser.add_argument("--length", type=float, required=True)
arg_parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
arg_parser.add_argument("--logdir", type=str, required=True)
arg_parser.add_argument("--trajectory", type=str, default="lift_and_drop")
arg_parser.add_argument("--motor", type=str, required=True)
arg_parser.add_argument("--kp", type=int, default=32)
arg_parser.add_argument("--vin", type=float, default=7.4)
arg_parser.add_argument("--id", type=int, required=True)
args = arg_parser.parse_args()

# 根据舵机型号选择 P 增益寄存器地址
if args.motor not in ADDR_P_GAIN_MAP:
    raise ValueError(f"不支持的舵机型号: {args.motor}，支持: {list(ADDR_P_GAIN_MAP.keys())}")
ADDR_P_GAIN = ADDR_P_GAIN_MAP[args.motor]
print(f"舵机型号: {args.motor}, P 增益寄存器地址: {ADDR_P_GAIN}")

os.makedirs(args.logdir, exist_ok=True)

if args.trajectory not in trajectories:
    raise ValueError(f"Unknown trajectory: {args.trajectory}")

motor_id = args.id

# ==================== 初始化 ftservo-python-sdk ====================
# 通信板：Feetech FT-URT2 USB 转 TTL
portHandler = PortHandler(args.port)

# 打开串口
if not portHandler.openPort():
    raise RuntimeError(f"无法打开串口 {args.port}")
if not portHandler.setBaudRate(1000000):
    raise RuntimeError("无法设置波特率 1000000")

# 创建 STS 系列舵机通信处理器
packetHandler = sms_sts(portHandler)


def check_result(result, error, msg=""):
    if result != COMM_SUCCESS:
        raise RuntimeError(f"{msg} 通信错误: {packetHandler.getTxRxResult(result)}")
    if error != 0:
        raise RuntimeError(f"{msg} 舵机错误: {packetHandler.getRxPacketError(error)}")


# 辅助：写 1 字节
def write1(addr, value):
    result, error = packetHandler.write1ByteTxRx(motor_id, addr, value)
    check_result(result, error, f"写寄存器 addr={addr}")

# 辅助：写 2 字节
def write2(addr, value):
    result, error = packetHandler.write2ByteTxRx(motor_id, addr, value)
    check_result(result, error, f"写寄存器 addr={addr}")

# 辅助：读 1 字节
def read1(addr):
    value, result, error = packetHandler.read1ByteTxRx(motor_id, addr)
    check_result(result, error, f"读寄存器 addr={addr}")
    return value

# 辅助：读 2 字节
def read2(addr):
    value, result, error = packetHandler.read2ByteTxRx(motor_id, addr)
    check_result(result, error, f"读寄存器 addr={addr}")
    return value


# 设置 P 增益
write1(ADDR_P_GAIN, args.kp)

# 使能扭矩
write1(SMS_STS_TORQUE_ENABLE, 1)

trajectory = trajectories[args.trajectory]

# 预热 1 秒
start = time.time()
while time.time() - start < 1.0:
    goal_position, torque_enable = trajectory(0)
    # 使用 SDK 自带的 ReadPos 做符号转换的逆运算来写位置
    # goal_position 是弧度，转成原始有符号值
    pos_value = int(goal_position * POS_SCALE)
    write2(SMS_STS_GOAL_POSITION_L, pos_value & 0xFFFF)
    # 每帧更新 P 增益
    write1(ADDR_P_GAIN, args.kp)
    time.sleep(0.01)

# ==================== 开始录制 ====================
start = time.time()
data = {
    "mass": args.mass,
    "length": args.length,
    "kp": args.kp,
    "vin": args.vin,
    "motor": args.motor,
    "trajectory": args.trajectory,
    "entries": [],
}


def read_data():
    # 终极优化：一条读指令，一次性读出所有需要的数据
    # 寄存器地址从 56 到 63，连续 8 字节：
    #   56-57: Present Position (2字节)
    #   58-59: Present Speed (2字节)
    #   60-61: Present Load / PWM Duty (2字节)
    #   62:    Present Voltage (1字节)
    #   63:    Present Temperature (1字节)
    all_data, result, error = packetHandler.readTxRx(
        motor_id, SMS_STS_PRESENT_POSITION_L, 8
    )
    check_result(result, error, "批量读取位置/速度/负载/电压/温度")

    # 位置：低字节在前
    pos_raw = all_data[0] | (all_data[1] << 8)
    pos_raw = packetHandler.scs_tohost(pos_raw, 15)  # 16位有符号转换
    position = pos_raw / POS_SCALE  # 原始值 → 弧度

    # 速度：低字节在前
    speed_raw = all_data[2] | (all_data[3] << 8)
    speed_raw = packetHandler.scs_tohost(speed_raw, 15)  # 16位有符号转换
    speed = speed_raw * SPEED_SCALE  # 原始值 → rad/s

    # 负载/PWM占空比：低字节在前，有符号
    # 单位：原始值 / 1000 = 占空比 [-1, 1]（负号表示反向）
    load_raw = all_data[4] | (all_data[5] << 8)
    load_raw = packetHandler.scs_tohost(load_raw, 15)  # 16位有符号转换
    load = load_raw / 1000.0  # 原始值 → 占空比 [-1, 1]

    # 电压：原始值 × 0.1 = V
    volts = all_data[6] * 0.1

    # 温度
    temp = float(all_data[7])

    return {
        "position": float(position),
        "speed": float(speed),
        "load": float(load),
        "input_volts": float(volts),
        "temp": float(temp),
    }


torque_enable = True
while time.time() - start < trajectory.duration:
    t = time.time() - start
    goal_position, new_torque_enable = trajectory(t)

    # 扭矩使能切换
    if new_torque_enable != torque_enable:
        write1(SMS_STS_TORQUE_ENABLE, 1 if new_torque_enable else 0)
        torque_enable = new_torque_enable
        time.sleep(0.001)

    # 发送目标位置
    if torque_enable:
        pos_value = int(goal_position * POS_SCALE)
        write2(SMS_STS_GOAL_POSITION_L, pos_value & 0xFFFF)
        time.sleep(0.001)

    # 读取数据
    t0 = time.time() - start
    entry = read_data()
    t1 = time.time() - start

    entry["timestamp"] = (t0 + t1) / 2.0
    entry["goal_position"] = goal_position
    entry["torque_enable"] = torque_enable
    data["entries"].append(entry)

# ==================== 录制结束，缓慢回零 ====================
goal_position = data["entries"][-1]["position"]
return_dt = 0.01
max_variation = return_dt * 1.0

while abs(goal_position) > 0:
    if goal_position > 0:
        goal_position = max(0, goal_position - max_variation)
    else:
        goal_position = min(0, goal_position + max_variation)

    pos_value = int(goal_position * POS_SCALE)
    write2(SMS_STS_GOAL_POSITION_L, pos_value & 0xFFFF)
    time.sleep(return_dt)

# 归位到零位
write2(SMS_STS_GOAL_POSITION_L, 0)
time.sleep(1)

# 关闭扭矩
write1(SMS_STS_TORQUE_ENABLE, 0)

# 关闭串口
portHandler.closePort()

# ==================== 保存数据 ====================
date = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
filename = f"{args.logdir}/{date}.json"
json.dump(data, open(filename, "w"))
