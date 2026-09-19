# Copyright 2026 ftservo

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
import sys
import numpy as np
import argparse
import time
from bam.trajectory import *

# Windows 系统提高定时器精度到 1ms
# 否则 time.sleep(0.001) 实际会等 15.6ms
if sys.platform == "win32":
    import ctypes
    winmm = ctypes.windll.winmm
    winmm.timeBeginPeriod(1)

# P 增益寄存器地址（不同舵机型号地址不同）
# STS3215: 地址 21 (0x15)
# HD1910:  地址 50 (0x32)
ADDR_P_GAIN_MAP = {
    "sts3215": 21,
    "hd1910": 50,
}

# 位置换算：
# 0 rad → 2048（中点）
# -π rad → 0
# +π rad → 4095
# 12 位有效精度，4096 计数 = 2π（一圈）
POS_COUNT_PER_REV = 4096.0

def pos_rad_to_raw(rad):
    """弧度 → 舵机原始位置值"""
    return int(POS_COUNT_PER_REV * (rad / (2 * np.pi) + 0.5))

def pos_raw_to_rad(raw):
    """舵机原始位置值 → 弧度"""
    return 2 * np.pi * ((raw / POS_COUNT_PER_REV) - 0.5)

# 速度单位换算：原始值 × 0.732 RPM → rad/s
# STS3215 和 HD1910 通用
# 1 RPM = 2π/60 rad/s
RPM_TO_RAD_S = 2 * np.pi / 60
SPEED_SCALE = 0.732 * RPM_TO_RAD_S  # 原始值 → rad/s

arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--mass", type=float, required=True)
arg_parser.add_argument("--arm-mass", type=float, default=0.0, help="摆臂自身质量 [kg]")
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


# 辅助：写 1 字节（带应答）
def write1(addr, value):
    result, error = packetHandler.write1ByteTxRx(motor_id, addr, value)
    check_result(result, error, f"写寄存器 addr={addr}")

# 辅助：写 1 字节（同步写，广播指令，舵机不应答，无残留）
def write1_only(addr, value):
    # 同步写：[ID, 数据]
    param = [motor_id, value & 0xFF]
    packetHandler.syncWriteTxOnly(addr, 1, param, 2)

# 辅助：写 2 字节（带应答）
def write2(addr, value):
    result, error = packetHandler.write2ByteTxRx(motor_id, addr, value)
    check_result(result, error, f"写寄存器 addr={addr}")

# 辅助：写 2 字节（同步写，广播指令，舵机不应答，无残留）
def write2_only(addr, value):
    # 同步写：[ID, 数据低字节, 数据高字节]
    param = [motor_id, value & 0xFF, (value >> 8) & 0xFF]
    packetHandler.syncWriteTxOnly(addr, 2, param, 3)

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


def read_data():
    # 一条读指令，一次性读出所有需要的数据
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

    # 位置：低字节在前 → 有符号转换 → 转弧度
    pos_raw = all_data[0] | (all_data[1] << 8)
    pos_raw = packetHandler.scs_tohost(pos_raw, 15)
    position = pos_raw_to_rad(pos_raw)

    # 速度：低字节在前
    speed_raw = all_data[2] | (all_data[3] << 8)
    speed_raw = packetHandler.scs_tohost(speed_raw, 15)
    speed = speed_raw * SPEED_SCALE

    # 负载/PWM占空比：低字节在前，有符号
    load_raw = all_data[4] | (all_data[5] << 8)
    load_raw = packetHandler.scs_tohost(load_raw, 15)
    load = load_raw / 1000.0  # 1000 = 100% PWM

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


try:
    trajectory = trajectories[args.trajectory]

    # 预热：写 P 增益和目标位置，最多重试 10 次，失败就报错退出
    # 写成功后等 1 秒稳定
    goal_position, torque_enable = trajectory(0)
    pos_value = pos_rad_to_raw(goal_position)

    max_retry = 10
    for i in range(max_retry):
        try:
            write1(ADDR_P_GAIN, args.kp)
            write2(SMS_STS_GOAL_POSITION_L, pos_value & 0xFFFF)
            break  # 都成功了就跳出
        except RuntimeError as e:
            if i == max_retry - 1:
                raise RuntimeError(f"预热失败，共尝试 {max_retry} 次：{e}") from e
            print(f"预热第 {i+1} 次失败：{e}，10ms 后重试...")
            time.sleep(0.01)

    print(f"预热完成：P 增益和目标位置已写入（第 {i+1} 次成功），等待 1 秒稳定")
    time.sleep(1)

    # ==================== 开始录制 ====================
    start = time.time()
    data = {
        "mass": args.mass,
        "arm-mass": args.arm_mass,
        "length": args.length,
        "kp": args.kp,
        "vin": args.vin,
        "motor": args.motor,
        "trajectory": args.trajectory,
        "entries": [],
    }


    torque_enable = True
    while time.time() - start < trajectory.duration:
        loop_start = time.time() - start
        t = loop_start
        goal_position, new_torque_enable = trajectory(t)

        # 扭矩使能切换（同步写，不应答）
        if new_torque_enable != torque_enable:
            write1_only(SMS_STS_TORQUE_ENABLE, 1 if new_torque_enable else 0)
            torque_enable = new_torque_enable

        # 发送目标位置（同步写，不应答）
        if torque_enable:
            pos_value = pos_rad_to_raw(goal_position)
            write2_only(SMS_STS_GOAL_POSITION_L, pos_value & 0xFFFF)

        # 读取数据
        t0 = time.time() - start
        entry = read_data()
        t1 = time.time() - start

        entry["timestamp"] = (t0 + t1) / 2.0
        entry["goal_position"] = goal_position
        entry["torque_enable"] = torque_enable
        data["entries"].append(entry)

        # 动态延时：保证每帧至少 1ms 周期
        loop_duration = time.time() - start - loop_start
        if loop_duration < 0.001:
            time.sleep(0.001 - loop_duration)

    # ==================== 录制结束，先延时再回零 ====================
    time.sleep(0.5)  # 等 0.5 秒，让舵机稳定

    # 读当前实际位置作为回零起点
    current = read_data()
    goal_position = current["position"]
    return_dt = 0.01
    max_variation = return_dt * 1.0  # 1 rad/s

    while abs(goal_position) > 0:
        if goal_position > 0:
            goal_position = max(0, goal_position - max_variation)
        else:
            goal_position = min(0, goal_position + max_variation)

        pos_value = pos_rad_to_raw(goal_position)
        write2_only(SMS_STS_GOAL_POSITION_L, pos_value & 0xFFFF)
        time.sleep(return_dt)

    # 归位到零位（中点 2048）
    write2_only(SMS_STS_GOAL_POSITION_L, pos_rad_to_raw(0))
    time.sleep(1)

finally:
    # 不管出不出错，都要关闭扭矩和串口
    try:
        write1_only(SMS_STS_TORQUE_ENABLE, 0)  # 关闭扭矩（只发不收）
    except Exception:
        pass  # 关失败也无所谓，反正串口要关了
    portHandler.closePort()  # 关闭串口

# ==================== 保存数据 ====================
date = datetime.datetime.now().strftime("%Y-%m-%d_%Hh%Mm%S")
filename = f"{args.logdir}/{date}.json"
with open(filename, "w") as f:
    json.dump(data, f)
