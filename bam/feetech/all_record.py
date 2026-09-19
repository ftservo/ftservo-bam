# Copyright 2026 ftservo

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at:

#     http://www.apache.org/licenses/LICENSE-2.0

import argparse
import sys
import time
import subprocess


arg_parser = argparse.ArgumentParser()
arg_parser.add_argument("--mass", type=float, required=True)
arg_parser.add_argument("--length", type=float, required=True)
arg_parser.add_argument("--motor", type=str, required=True)
arg_parser.add_argument("--port", type=str, default="/dev/ttyUSB0")
arg_parser.add_argument("--id", type=int, required=True)
arg_parser.add_argument("--logdir", type=str, required=True)
arg_parser.add_argument("--vin", type=float, default=7.4)
args = arg_parser.parse_args()

# 不同舵机型号适用的 P 增益值（0-255 范围）
KP_MAP = {
    "sts3215": [4, 8, 16, 32],
    "hd1910": [4, 8, 16, 32],  # 同协议，P 范围一样
}

# 激励轨迹（覆盖不同摩擦工况）
trajectories = ["sin_sin", "lift_and_drop", "up_and_down", "sin_time_square"]

# 检查舵机型号
if args.motor not in KP_MAP:
    raise ValueError(f"不支持的舵机型号: {args.motor}，支持: {list(KP_MAP.keys())}")

kps = KP_MAP[args.motor]
print(f"舵机型号: {args.motor}, P 增益列表: {kps}")

# 基础命令（用 sys.executable 确保用当前 Python 环境）
command_base = [
    sys.executable,
    "-m", "bam.feetech.record",
    "--mass", str(args.mass),
    "--length", str(args.length),
    "--port", args.port,
    "--logdir", args.logdir,
    "--motor", args.motor,
    "--id", str(args.id),
    "--vin", str(args.vin),
]


for kp in kps:
    for trajectory in trajectories:
        sentence = f"Kp {kp}, trajectory {trajectory.replace('_', ' ')}"
        print(sentence)

        # 完整命令
        command = command_base + [
            "--kp", str(kp),
            "--trajectory", trajectory,
        ]

        # 用 subprocess.run 执行，失败了会抛错
        subprocess.run(command, check=True)

        # sin_time_square 后等待系统稳定
        if trajectory == "sin_time_square":
            time.sleep(3)
