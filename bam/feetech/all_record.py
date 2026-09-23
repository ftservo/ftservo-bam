# Copyright 2026 ftservo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import sys
import time
import subprocess

from scservo_sdk import PortHandler, COMM_SUCCESS
from scservo_sdk.sms_sts import sms_sts, SMS_STS_PRESENT_TEMPERATURE, SMS_STS_PRESENT_VOLTAGE
from bam.trajectory import trajectories as ALL_TRAJECTORIES


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--mass", type=float, required=True)
arg_parser.add_argument("--arm-mass", type=float, default=0.0, help="摆臂自身质量 [kg]")
arg_parser.add_argument("--length", type=float, required=True)
arg_parser.add_argument("--motor", type=str, required=True)
arg_parser.add_argument("--port", type=str, default="COM18")
arg_parser.add_argument("--id", type=int, required=True)
arg_parser.add_argument("--logdir", type=str, required=True)
arg_parser.add_argument("--vin", type=float, default=7.4)
arg_parser.add_argument("--kps", type=str, default="", help="只跑指定的 kp，逗号分隔，如 4,8；留空=全跑")
arg_parser.add_argument("--trajectories", type=str, default="", help="只跑指定的轨迹，逗号分隔，如 up_and_down,half_sine；留空=跑默认 4 条")
arg_parser.add_argument("--cooldown-temp", type=float, default=45.0, help="每块开跑前等舵机降到该温度以下 [°C]；设为 0 关闭冷却等待")
arg_parser.add_argument("--cooldown-timeout", type=float, default=900.0, help="冷却等待上限 [秒]，超时则退出")
arg_parser.add_argument("--dry-run", action="store_true", help="只打印将要执行的命令，不真的跑（不碰舵机）")
args = arg_parser.parse_args()

# 不同舵机型号适用的 P 增益值（0-255 范围）
KP_MAP = {
    "sts3215": [4, 8, 16, 32, 64],
    "hd1910": [4, 8, 16, 32, 64],  # 同协议，P 范围一样
}

# 激励轨迹（覆盖不同摩擦工况）—— 批量默认就跑这 4 条
# 注意：轨迹库里还有 half_sine / steps / nothing，可以用 --trajectories 手动指定
trajectories = ["sin_sin", "lift_and_drop", "up_and_down", "sin_time_square"]

# 检查舵机型号
if args.motor not in KP_MAP:
    raise ValueError(f"不支持的舵机型号: {args.motor}，支持: {list(KP_MAP.keys())}")

kps = KP_MAP[args.motor]
if args.kps:
    try:
        kps = [int(x) for x in args.kps.replace(" ", "").split(",") if x]
    except ValueError:
        raise SystemExit(f"--kps 格式错误：{args.kps}（应为逗号分隔整数，如 4,8,16）")
    unknown = [k for k in kps if k not in KP_MAP[args.motor]]
    if unknown:
        raise SystemExit(f"--kps 中的 {unknown} 不在 {args.motor} 的可用列表 {KP_MAP[args.motor]} 内")
print(f"舵机型号: {args.motor}, P 增益列表: {kps}")

# 只跑指定轨迹（留空=跑上面那 4 条默认）。白名单用轨迹库全集，所以 half_sine 也能选
if args.trajectories:
    selected_trajectories = [t.strip() for t in args.trajectories.split(",") if t.strip()]
    unknown = [t for t in selected_trajectories if t not in ALL_TRAJECTORIES]
    if unknown:
        raise SystemExit(f"--trajectories 中的 {unknown} 不在可用轨迹 {list(ALL_TRAJECTORIES)} 内")
else:
    selected_trajectories = list(trajectories)
print(f"轨迹列表: {selected_trajectories}")


def _safe_close(port_handler):
    """串口可能压根没打开成功（连 ser 对象都没建），关失败直接忽略。"""
    try:
        port_handler.closePort()
    except Exception:
        pass


def read_servo_status():
    """纯只读：读温度 / 电压 / 错误标志。不写任何寄存器、不使能扭矩。"""
    port_handler = PortHandler(args.port)
    try:
        try:
            if not port_handler.openPort() or not port_handler.setBaudRate(1000000):
                raise RuntimeError(f"无法打开串口 {args.port} 或设置波特率 1000000 失败")
        except RuntimeError:
            raise
        except Exception as e:
            raise RuntimeError(f"无法打开串口 {args.port}：{e}（端口被占用或不存在：查 FD 软件、残留 python 进程、USB 连接）") from e

        packet_handler = sms_sts(port_handler)
        temp, r1, e1 = packet_handler.read1ByteTxRx(args.id, SMS_STS_PRESENT_TEMPERATURE)
        volt, r2, e2 = packet_handler.read1ByteTxRx(args.id, SMS_STS_PRESENT_VOLTAGE)
        if r1 != COMM_SUCCESS and r2 != COMM_SUCCESS:
            raise RuntimeError("读取温度/电压通信失败（串口异常或接线松动）")
        return float(temp), float(volt) * 0.1, int(e1) | int(e2)
    finally:
        _safe_close(port_handler)


def wait_for_cooldown():
    """等舵机降温到 --cooldown-temp 以下再开始下一块。等待期间只读，不动舵机。"""
    target = args.cooldown_temp
    deadline = time.time() + args.cooldown_timeout
    err_streak = 0
    while True:
        try:
            temp, volt, err = read_servo_status()
            if err:
                err_streak += 1
                print(f"[冷却] 温度 {temp:.0f}°C  电压 {volt:.1f}V  !! 舵机错误标志 = {err}（连续 {err_streak}/3）")
                if err_streak >= 3:
                    raise SystemExit("舵机持续报错（多半是锁存的过流/欠压标志），请断电 10 秒重上电后再跑")
            else:
                err_streak = 0
                print(f"[冷却] 温度 {temp:.0f}°C  电压 {volt:.1f}V  错误 无")
                if temp < target:
                    return
        except Exception as e:
            print(f"[冷却] 读取失败：{e}（10 秒后重试）")
        if time.time() > deadline:
            raise SystemExit(f"冷却等待超过 {args.cooldown_timeout:.0f} 秒仍未降到 {target:.0f}°C，请检查散热后重跑")
        time.sleep(10)

# 基础命令（用 sys.executable 确保用当前 Python 环境）
command_base = [
    sys.executable,
    "-m", "bam.feetech.record",
    "--mass", str(args.mass),
    "--arm-mass", str(args.arm_mass),
    "--length", str(args.length),
    "--port", args.port,
    "--logdir", args.logdir,
    "--motor", args.motor,
    "--id", str(args.id),
    "--vin", str(args.vin),
]


recorded = 0
for block_idx, kp in enumerate(kps, 1):
    # 每块开跑前等降温（dry-run 跳过）
    if args.cooldown_temp > 0 and not args.dry_run:
        print(f"\n===== 第 {block_idx}/{len(kps)} 块（kp={kp}）：等待降温到 {args.cooldown_temp:.0f}°C 以下 =====")
        wait_for_cooldown()
        print(f"温度达标，开始录制 kp={kp}\n")

    for trajectory in selected_trajectories:
        sentence = f"Kp {kp}, trajectory {trajectory.replace('_', ' ')}"
        print(sentence)

        # 完整命令
        command = command_base + [
            "--kp", str(kp),
            "--trajectory", trajectory,
        ]

        if args.dry_run:
            print("  [dry-run] " + " ".join(command))
            recorded += 1
            continue

        # 用 subprocess.run 执行，失败了会抛错
        subprocess.run(command, check=True)
        recorded += 1

        # sin_time_square 后等待系统稳定
        if trajectory == "sin_time_square":
            time.sleep(3)

print(f"\n全部完成：共录制 {recorded} 条（logdir={args.logdir}）")
