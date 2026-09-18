# `bam/feetech/` 源码解读

Feetech（飞特）模块是 BAM 框架中对 **Feetech STS3215 总线舵机** 的支持实现，包含仿真模型和硬件数据采集两部分。

---

## 目录结构总览

```
bam/feetech/
├── __init__.py      # 包标识（仅许可证声明）
├── actuator.py      # 【核心】STS3215 仿真模型（继承自 VoltageControlledActuator）
├── record.py        # 【硬件】单条轨迹录制脚本
└── all_record.py    # 【硬件】批量录制脚本（遍历所有 P 增益 × 轨迹）
```

---

## 一、`actuator.py` — STS3215 仿真模型

### 类定义：`STS3215Actuator`

继承自 `VoltageControlledActuator`（电压控制型执行器基类），专门模拟 Feetech STS3215（7.4V 版本）舵机的行为。

### 1.1 构造函数参数

```python
super().__init__(
    testbench_class,
    vin=7.4,           # 供电电压 7.4V
    kp=32,             # 固件 P 增益（Feetech 固件单位）
    error_gain=0.166,  # 误差增益（示波器实测标定）
    max_pwm=0.97,      # 最大占空比（示波器实测）
)
```

**关键设计：**
- `error_gain=0.166`：这是一个**硬件标定常数**，由示波器实测确定，作用是将「位置误差 × 固件 KP」转换为 PWM 占空比
- `max_pwm=0.97`：最大占空比限制，也是示波器实测值（理论上可以假设 1.0，但实际留了 3% 余量）

### 1.2 速率限制器（STS3215 的特殊之处）

STS3215 固件有一个**关键特性**：它不直接跟踪目标位置，而是在内部维护一个**速率受限的目标位置**：

```python
self.q_target_smooth: ArrayLike | None = None
```

**工作原理：**
1. 上位机发送目标位置 `q_target`
2. 固件内部的 `q_target_smooth` 以 `max_velocity` 为上限向 `q_target` 靠近
3. P 控制器作用在 `q_target_smooth` 与当前位置 `q` 的误差上

这就是为什么这个类标记为 `stateful = True`——控制器是有状态的，必须按时间步顺序调用。

### 1.3 `compute_control()` 核心控制逻辑

```python
def compute_control(self, q_target, q, dq, dt):
    # 1. 初始化或重置内部目标
    if q_target_smooth is None:
        q_target_smooth = q  # 上电时内部目标等于当前位置
    
    # 2. 速率限制：内部目标向外部目标靠近，但不超过 max_velocity * dt
    max_step = self.model.max_velocity.value * dt
    self.q_target_smooth = clamp(
        q_target,
        q_target_smooth - max_step,
        q_target_smooth + max_step,
    )
    
    # 3. P 控制：计算占空比
    duty_cycle = (
        (self.q_target_smooth - q)  # 用内部平滑后的目标计算误差
        * self.kp
        * self.error_gain
        * self.model.error_gain_ratio.value
    )
    duty_cycle = clamp(duty_cycle, -self.max_pwm, self.max_pwm)
    
    # 4. 输出电压 = 供电电压 × 占空比
    return self.vin * duty_cycle
```

**这个设计非常重要：** 它准确模拟了 STS3215 固件的行为——上位机发的目标位置不是直接跟踪的，而是先经过一个速度限制器。如果不模拟这个特性，仿真会和真机差很多。

### 1.4 待辨识参数

`initialize()` 方法声明了需要优化的参数及其初始猜测和搜索范围：

| 参数 | 初始值 | 搜索范围 | 物理意义 |
|------|--------|---------|---------|
| `kt` | 0.7845 | [0.05, 2.5] | 扭矩常数 [Nm/A]（手册标称 8 kg·cm/A） |
| `error_gain_ratio` | 1.0 | [0.1, 10.0] | 误差增益修正系数 |
| `R` | 2.0 | [0.1, 10.0] | 电机电阻 [Ω] |
| `armature` | 0.0001 | [1e-5, 0.04] | 等效转动惯量 [kg·m²] |
| `q_offset` | 0 | [-0.2, 0.2] | 安装位置偏移 [rad] |
| `max_velocity` | 3400×2π/4096 | [0.1×default, 10×default] | 固件最大速度限制 |

### 1.5 重置机制（向量化支持）

`reset()` 和 `_to_reset_env_ids` 的设计是为了支持 **mjlab/GPU 向量化仿真**：
- 当一批环境中的某些环境被重置时，它们的内部 `q_target_smooth` 需要跟随传送后的位置
- 这个机制确保了 GPU 批量仿真时的状态正确性

---

## 二、`record.py` — 单条轨迹硬件录制

### 功能概述

连接真实 STS3215 舵机，运行一条预定义轨迹，记录位置/速度/电压等数据到 JSON 文件。

### 2.1 命令行参数

```bash
python -m bam.feetech.record \
    --mass 0.5          # 配重质量 [kg]
    --length 0.15       # 摆臂长度 [m]
    --motor sts3215     # 电机名称标识
    --port /dev/ttyACM0 # 串口设备
    --id 1              # 舵机 ID
    --kp 32             # P 增益
    --vin 7.4           # 供电电压
    --trajectory lift_and_drop  # 轨迹名称
    --logdir data_raw   # 输出目录
```

### 2.2 硬件通信层

```python
from pypot.feetech import FeetechSTS3215IO

io = FeetechSTS3215IO("/dev/ttyACM0")
io.set_mode({1: 0})  # 位置控制模式
```

使用 `pypot` 库的 Feetech 驱动与舵机通信。

### 2.3 录制主循环

```python
while time.time() - start < trajectory.duration:
    t = time.time() - start
    goal_position, torque_enable = trajectory(t)  # 生成轨迹点
    
    # 使能/失能扭矩切换
    if new_torque_enable != torque_enable:
        if new_torque_enable:
            io.enable_torque([1])
        else:
            io.disable_torque([1])
    
    # 发送目标位置（弧度 → 角度转换）
    io.set_goal_position({1: np.rad2deg(goal_position)})
    
    # 读取传感器数据
    position = np.deg2rad(io.get_present_position([1])[0])
    speed = np.deg2rad(io.get_present_speed([1])[0])
    volts = io.get_present_voltage([1])[0] * 0.1  # 原始值 × 0.1 = V
```

### 2.4 数据格式

每条记录生成一个 JSON 文件：

```json
{
  "mass": 0.5,
  "length": 0.15,
  "kp": 32,
  "vin": 7.4,
  "motor": "sts3215",
  "trajectory": "lift_and_drop",
  "entries": [
    {
      "timestamp": 0.0077,
      "position": 0.0015,
      "speed": 0.024,
      "load": 0.0,
      "input_volts": 7.4,
      "goal_position": 0.0,
      "torque_enable": true
    }
  ]
}
```

### 2.5 录制结束后的归位

轨迹播放完成后，舵机会**缓慢回零**（而不是直接跳回），防止突然断电或急停造成机械冲击：

```python
# 缓慢回到零位，每步移动一点
while abs(goal_position) > 0:
    goal_position = max(0, goal_position - max_variation)
    io.set_goal_position({1: np.rad2deg(goal_position)})
    time.sleep(return_dt)
```

---

## 三、`all_record.py` — 批量录制脚本

### 功能概述

自动化批量录制：自动遍历**所有 P 增益 × 所有轨迹**的组合，完成完整的数据采集。

### 3.1 配置

```python
kps = [4, 8, 16, 32]  # 4 个 P 增益值
trajectories = [
    "brutal",
    "sin_sin",
    "lift_and_drop",
    "up_and_down",
    "sin_time_square"
]  # 5 种轨迹
```

总共：4 × 5 = **20 段记录**（单组质量/长度配置下）

### 3.2 执行流程

```python
for kp in kps:
    for trajectory in trajectories:
        # 语音播报当前录制阶段（可选）
        if args.speak:
            from gtts import gTTS
            myobj = gTTS(text=f"Kp {kp}, trajectory {trajectory}", lang="en")
            myobj.save("/tmp/message.mp3")
            os.system("mpg321 /tmp/message.mp3")
        
        # 调用单条记录脚本
        command = f"python3 -m bam.feetech.record --kp {kp} --trajectory {trajectory} ..."
        os.system(command)
```

### 3.3 设计特点

- **语音提示**：录制过程中用 TTS 播报当前阶段，方便人在旁边观察硬件
- **简单调用**：通过 `os.system()` 调用 `record.py`，而不是直接 import——这样每条记录都是独立进程，避免状态污染
- **间隔等待**：`sin_time_square` 轨迹后等待 3 秒，让系统稳定

---

## 四、Feetech STS3215 的关键特性总结

| 特性 | 说明 | 在代码中的体现 |
|------|------|--------------|
| **固件速率限制** | 内部目标位置有最大速度限制 | `q_target_smooth` + `max_velocity` 参数 |
| **电压控制型** | 固件输出 PWM 电压，不是电流 | 继承 `VoltageControlledActuator` |
| **7.4V 供电** | 标称工作电压 7.4V | `vin=7.4` |
| **P 增益单位** | 固件 KP 范围 0-255 | `kp=32`（默认值） |
| **角度通信** | 上位机指令单位是度 | `np.rad2deg()` 转换 |
| **电压读取** | 传感器返回值 × 0.1 = 伏特 | `volts * 0.1` |

---

## 五、与 BAM 框架的集成关系

```
                    ┌─────────────────────────┐
                    │   bam/feetech/         │
                    │  ┌───────────────────┐  │
                    │  │ STS3215Actuator   │  │ ← 仿真时使用（参数拟合、MuJoCo）
                    │  │ (继承 Voltage...   │  │
                    │  └───────────────────┘  │
                    │  ┌───────────────────┐  │
                    │  │ record.py         │  │ ← 硬件采集时使用
                    │  │ all_record.py     │  │
                    │  └───────────────────┘  │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        bam/actuator.py   bam/fit.py    bam/mujoco.py
        (基类定义)       (参数拟合)     (仿真集成)
```

**仿真路径：** `STS3215Actuator` → 被 `bam/fit.py` 调用拟合参数 → 参数 JSON 传给 `bam/mujoco.py` 做 MuJoCo 仿真

**硬件路径：** `record.py` → 连接真实舵机采集数据 → 数据交给 `bam/process.py` 预处理 → 再交给 `bam/fit.py` 拟合

---

## 六、与 Dynamixel 实现的对比要点

Feetech STS3215 和 Dynamixel 舵机的主要区别：

| 对比项 | Feetech STS3215 | Dynamixel MX/XL |
|--------|----------------|-----------------|
| 控制方式 | 电压/PWM 控制 | 电压/PWM 控制 |
| 速率限制 | **有**（固件内部限速） | 无（直接跟踪目标） |
| 通信库 | `pypot.feetech` | `dynamixel_sdk` |
| 供电电压 | 7.4V | 12V |
| P 增益范围 | 0-255 | 0-1023 |
| 扭矩反馈 | 不支持（load=0） | 支持 |

最大的区别就是 **STS3215 固件有内部速率限制器**，这也是 `actuator.py` 中 `q_target_smooth` 状态变量存在的原因——这是 STS3215 特有的固件行为，必须在仿真中复现才能得到准确模型。
