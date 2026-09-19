# `bam/feetech/` 源码解读（修改后完整版）

> 基于修改后的代码：去除速率限制 + ftservo-python-sdk 官方驱动
> 硬件支持：Feetech STS3215 / HD1910 7.4V 总线舵机

---

## 目录结构总览

```
bam/feetech/
├── __init__.py      # 包标识（仅许可证声明）
├── actuator.py      # 【核心】仿真模型（STS3215Actuator + HD1910Actuator）
├── record.py        # 【硬件】单条轨迹录制（通用，STS3215/HD1910 共用）
└── all_record.py    # 【硬件】批量录制（遍历 P 增益 × 轨迹）
```

---

## 一、`actuator.py` — 仿真模型

文件中包含两个舵机类，都继承自 `VoltageControlledActuator`（电压控制型执行器基类）。

---

### 1.1 类：`STS3215Actuator`

模拟 Feetech STS3215 舵机行为。

**修改要点：** 已去除固件内部速率限制器，直接跟踪目标位置。

#### 构造函数参数

```python
def __init__(self, testbench_class: Testbench):
    super().__init__(
        testbench_class,
        vin=7.4,           # 供电电压 7.4V
        kp=32,             # 固件 P 增益（Feetech 固件单位 0-255）
        # 误差增益：将位置误差 × 固件 KP 转换为 PWM 占空比
        # 标定条件：KP=1 时，4000 计数（≈352°）误差对应 100% PWM
        error_gain=0.163,
        max_pwm=0.97,      # 最大占空比（留 3% 余量）
    )
```

#### error_gain 标定详解

**标定条件：**
- KP = 1
- 位置误差 = 4000 计数（舵机原始单位）
- 输出 PWM = 100%（满量程）

**计算过程：**
```
位置换算：4096 计数 = 2π → POS_SCALE = 4096/(2π) ≈ 651.9 计数/rad

误差(弧度) = 4000 / 651.9 ≈ 6.14 rad = 352°

error_gain = 1.0 / (6.14 × 1) ≈ 0.163
```

**反向验证：**
```
KP=1，4000 计数误差 → 6.14 × 1 × 0.163 = 1.0 = 100% PWM ✓
```

**KP=32 时满量程误差：**
- 计数：4000 / 32 = 125 计数
- 角度：≈ **11°**

#### 待辨识参数（`initialize()`）

```python
def initialize(self):
    self.model.kt = Parameter(0.784532, 0.05, 2.5)    # 扭矩常数 [Nm/A]
    self.model.error_gain_ratio = Parameter(1.0, 0.1, 10.0)  # 误差增益修正
    self.model.R = Parameter(2.0, 0.1, 10.0)           # 电机电阻 [Ω]
    self.model.armature = Parameter(0.0001, 0.00001, 0.04)  # 等效惯量 [kg·m²]
    self.model.q_offset = Parameter(0, -0.2, 0.2)      # 安装位置偏移 [rad]
```

| 参数 | 初始值 | 下限 | 上限 | 物理意义 |
|------|--------|------|------|---------|
| `kt` | 0.785 | 0.05 | 2.5 | 扭矩常数 [Nm/A] |
| `error_gain_ratio` | 1.0 | 0.1 | 10.0 | error_gain 修正系数 |
| `R` | 2.0 | 0.1 | 10.0 | 电机电阻 [Ω] |
| `armature` | 0.0001 | 1e-5 | 0.04 | 等效转动惯量 [kg·m²] |
| `q_offset` | 0.0 | -0.2 | 0.2 | 安装位置偏移 [rad] |

> 注：已删除原 `max_velocity` 参数（速率限制已去除）。

---

### 1.2 类：`HD1910Actuator`

模拟 Feetech HD1910 金属齿轮数字舵机行为。

**硬件规格（hd1910.MD）：**
- 硬件版本：10.31 (HDS1910)
- 固件版本：3.46
- 额定电压：7.4V
- 空载速度：150 × 0.732 = 109.8 rpm @7.4V
- 位置精度：12 位（4096 计数/圈）
- 中位位置：2048
- 死区宽度：≤0.088°

**与 STS3215 的关系：**
- 通信协议完全兼容（都是 SMS/STS 系列）
- 大部分寄存器地址相同，但 **P 增益地址不同**
- 单位换算完全相同（12位位置、0.732 RPM 速度）
- 区别：P 增益地址 + 物理参数（扭矩常数、电阻、惯量）

#### 构造函数参数

```python
def __init__(self, testbench_class: Testbench):
    super().__init__(
        testbench_class,
        vin=7.4,           # 供电电压 7.4V（和 STS3215 一样）
        kp=32,             # 固件 P 增益
        # 误差增益：与 STS3215 同协议，相同标定值
        # 标定条件：KP=1 时，4000 计数误差对应 100% PWM
        error_gain=0.163,
        max_pwm=0.97,
    )
```

#### 待辨识参数（`initialize()`）

```python
def initialize(self):
    # 扭矩常数 [Nm/A]，根据空载速度估算：kt = V / ω = 7.4V / 11.5 rad/s ≈ 0.64
    self.model.kt = Parameter(0.64, 0.05, 2.5)
    self.model.error_gain_ratio = Parameter(1.0, 0.1, 10.0)
    self.model.R = Parameter(5.0, 0.5, 20.0)          # 电阻
    self.model.armature = Parameter(0.00005, 0.000005, 0.02)  # 惯量
    self.model.q_offset = Parameter(0, -0.2, 0.2)
```

| 参数 | 初始值 | 下限 | 上限 | 说明 |
|------|--------|------|------|------|
| `kt` | 0.64 | 0.05 | 2.5 | 空载速度法估算（109.8 rpm） |
| `error_gain_ratio` | 1.0 | 0.1 | 10.0 | 误差增益修正系数 |
| `R` | 5.0 | 0.5 | 20.0 | 电机电阻 |
| `armature` | 0.00005 | 5e-6 | 0.02 | 转子惯量 |
| `q_offset` | 0.0 | -0.2 | 0.2 | 安装位置偏移 |

> 注：以上初始值为经验估计值，BAM 拟合时会自动校准到真实值。

---

### 1.3 共用控制逻辑：`compute_control()`

两个类的 `compute_control()` 完全相同：

```python
def compute_control(self, q_target, q, dq, dt):
    duty_cycle = (
        (q_target - q)       # 目标位置误差（弧度）
        * self.kp            # 固件 P 增益
        * self.error_gain    # 硬件标定误差增益
        * self.model.error_gain_ratio.value  # 辨识修正系数
    )
    duty_cycle = self.backend.clamp(duty_cycle, -self.max_pwm, self.max_pwm)
    self.duty_cycle = duty_cycle
    return self.vin * duty_cycle  # 输出电压 = 供电电压 × 占空比
```

**控制流程：**
```
目标位置 q_target
     ↓
误差 = q_target - q（弧度）
     ↓
占空比 = 误差 × KP × error_gain × ratio
     ↓
clamp 到 [-0.97, +0.97]
     ↓
输出电压 = 7.4V × 占空比
```

**关键特性：**
- ✅ **无内部速率限制**：P 控制器直接作用在「上位机目标 - 当前位置」的误差上
- ✅ **无状态**：不需要 `q_target_smooth` 中间变量
- ✅ **支持向量化**：可批量处理多个环境（GPU 仿真）

---

### 1.4 两个舵机参数对比

| 参数 | STS3215 | HD1910 | 说明 |
|------|---------|--------|------|
| 供电电压 | 7.4V | 7.4V | 相同 |
| 通信协议 | SMS/STS | SMS/STS | 相同 |
| 位置精度 | 12 位（4096计数/圈） | 12 位（4096计数/圈） | 相同 |
| 速度单位 | 0.732 RPM/单位 | 0.732 RPM/单位 | 相同 |
| **P 增益地址** | **21 (0x15)** | **50 (0x32)** | **不同！** |
| 其他寄存器 | 相同 | 相同 | 位置/速度/电压/温度地址相同 |
| `error_gain` | 0.163（已标定） | 0.163（同协议，相同值） | 相同 |
| `kt`（初始值） | 0.785 Nm/A | 0.64 Nm/A | 空载速度法估算 |
| `R`（初始值） | 2.0 Ω | 5.0 Ω | 金属齿轮舵机电阻 |
| `armature`（初始值） | 0.0001 kg·m² | 0.00005 kg·m² | 金属齿轮舵机惯量 |
| 舵机版本 | - | 10.31 | HD1910 硬件版本 |

---

## 二、`record.py` — 硬件数据采集

### 功能概述

连接真实舵机，运行预定义轨迹，记录位置/速度/电压等数据到 JSON 文件。

**通信库：** `ftservo-python-sdk`（官方低层 SDK，导入名 `scservo_sdk`）
**通信板：** Feetech FT-URT2 USB 转 TTL 串口通信板

**说明：** STS3215 和 HD1910 的硬件代码基本通用，只有 P 增益寄存器地址不同，根据 `--motor` 参数自动选择。

---

### 2.1 硬件参数与单位换算

```python
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
RPM_TO_RAD_S = 2 * np.pi / 60     # ≈ 0.1047 rad/s per RPM
SPEED_SCALE = 0.732 * RPM_TO_RAD_S  # ≈ 0.0767 rad/s 每单位
```

**硬件规格总结：**

| 参数 | STS3215 | HD1910 | 说明 |
|------|---------|--------|------|
| 位置精度 | 12 位 | 12 位 | 4096 计数/圈（≈ 0.088°/步） |
| 速度单位 | 0.732 RPM/单位 | 0.732 RPM/单位 | 速度寄存器原始值的单位 |
| 电压单位 | 0.1 V/单位 | 0.1 V/单位 | 电压寄存器原始值的单位 |
| 供电电压 | 7.4 V | 7.4 V | 标称工作电压 |
| 波特率 | 1,000,000 bps | 1,000,000 bps | 高速串口通信 |
| **P 增益地址** | **21 (0x15)** | **50 (0x32)** | **不同！** |

---

### 2.2 寄存器地址映射

| 功能 | STS3215 地址 | HD1910 地址 | 常量名 | 读写 |
|------|-------------|------------|--------|------|
| **P 增益** | **21 (0x15)** | **50 (0x32)** | `ADDR_P_GAIN`（自动选择） | RW |
| 扭矩使能 | 40 | 40 | `SMS_STS_TORQUE_ENABLE` | RW |
| 目标位置（低字节） | 42 | 42 | `SMS_STS_GOAL_POSITION_L` | RW |
| 当前位置（低字节） | 56 | 56 | `SMS_STS_PRESENT_POSITION_L` | R |
| 当前速度（低字节） | 58 | 58 | `SMS_STS_PRESENT_SPEED_L` | R |
| **当前负载/PWM占空比** | **60** | **60** | —（批量读取解析） | R |
| 当前电压 | 62 | 62 | `SMS_STS_PRESENT_VOLTAGE` | R |
| 当前温度 | 63 | 63 | `SMS_STS_PRESENT_TEMPERATURE` | R |

> 注：地址 56~63 连续 8 字节，可一次批量读取，见 2.6 节。

---

### 2.3 SDK 初始化流程

```python
from scservo_sdk import PortHandler, COMM_SUCCESS
from scservo_sdk.sms_sts import sms_sts

# 1. 创建串口处理器
portHandler = PortHandler("/dev/ttyUSB0")
portHandler.openPort()
portHandler.setBaudRate(1000000)

# 2. 创建 STS 系列协议处理器（STS3215 和 HD1910 通用）
packetHandler = sms_sts(portHandler)

# 3. 辅助函数封装
def write1(addr, value):
    result, error = packetHandler.write1ByteTxRx(motor_id, addr, value)
    check_result(result, error)

def read1(addr):
    value, result, error = packetHandler.read1ByteTxRx(motor_id, addr)
    check_result(result, error)
    return value
```

**SDK 核心 API：**
- `PortHandler` — 串口管理（openPort / setBaudRate / closePort）
- `sms_sts(portHandler)` — STS 系列协议处理器（兼容 STS3215 和 HD1910）
- 读写方法：`write1ByteTxRx(id, addr, val)` / `read2ByteTxRx(id, addr)`
- 便捷方法：`ReadPos(id)` / `ReadSpeed(id)`（自动处理符号转换）

---

### 2.4 命令行参数

```bash
python -m bam.feetech.record `
    --mass 0.5          # 配重质量 [kg]（必填）
    --arm-mass 0.0      # 摆臂自身质量 [kg]（可选，默认 0.0）
    --length 0.15       # 摆臂长度 [m]（必填）
    --port /dev/ttyUSB0 # 串口设备（默认 /dev/ttyUSB0）
    --logdir data_raw   # 输出目录（必填）
    --trajectory sin_time_square  # 轨迹名称
    --motor hd1910      # 电机名称标识（必填，决定 P 增益地址 + 数据标签）
    --kp 32             # P 增益（默认 32）
    --vin 7.4           # 供电电压（默认 7.4V）
    --id 1              # 舵机 ID（必填）
```

> 注意：`--motor` 不仅是数据标签，还会影响 P 增益寄存器地址的选择：
> - `sts3215` → 写地址 21
> - `hd1910` → 写地址 50
>
> `--arm-mass` 是摆臂自身质量，用于 BAM 框架计算总惯量和重力偏置。如果摆臂很轻可以不填（默认 0.0）。

---

### 2.5 录制主流程

```
初始化 → 预热（写成功就停，最多试10次）→ 轨迹播放录制 → 延时0.5s稳定 → 读当前位置回零 → 关闭扭矩+串口 → 保存 JSON
```

**预热阶段：**
- 写 P 增益、目标位置（带应答确认）
- 最多重试 10 次，每次间隔 10ms
- 任何一次失败都打印具体原因，继续重试
- 10 次都失败就报错退出
- 写成功后等 1 秒稳定，然后开始录制

**录制结束回零流程：**
1. 录制结束后先等 **0.5 秒**，让舵机稳定
2. **读舵机当前实际位置**作为回零起点（不是用记录的最后一帧位置）
3. 以 **1 rad/s** 速度缓慢回零到 0（中点 2048）
4. 到达 0 后，再等 1 秒稳定
5. finally 块里关闭扭矩、关闭串口

**主循环：**
```python
while time.time() - start < trajectory.duration:
    loop_start = time.time() - start
    goal_position, new_torque_enable = trajectory(t)  # 生成轨迹点
    
    # 扭矩使能切换（同步写，广播指令，舵机不应答）
    if new_torque_enable != torque_enable:
        write1_only(SMS_STS_TORQUE_ENABLE, 1 if new_torque_enable else 0)
        torque_enable = new_torque_enable
    
    # 发送目标位置（同步写，广播指令，舵机不应答）
    if torque_enable:
        pos_value = pos_rad_to_raw(goal_position)
        write2_only(SMS_STS_GOAL_POSITION_L, pos_value & 0xFFFF)
    
    # 读取数据
    entry = read_data()
    data["entries"].append(entry)
    
    # 动态延时：保证每帧至少 1ms 周期
    loop_duration = time.time() - start - loop_start
    if loop_duration < 0.001:
        time.sleep(0.001 - loop_duration)
```

**为什么用同步写（而不是普通 TxOnly）：**
- 普通 TxOnly 写指令后，舵机仍然会返回应答包
- FT-URT2 通信板有返回延时，应答包会残留到下一次读操作里，导致读数据出错（HD1910 读 8 字节只返回 1 字节）
- **同步写（syncWriteTxOnly）是广播指令（ID=0xFE），舵机完全不应答**，从根本上消除残留问题

**写入方式对比：**
| 阶段 | 写入方式 | 说明 |
|------|---------|------|
| 预热阶段 | 带应答（write1/write2） | 确认写入成功 |
| 录制主循环 | **同步写广播（write1_only/write2_only）** | 舵机不应答，无残留，最快 |
| 回零阶段 | **同步写广播（write2_only）** | 不需要确认 |
| 关闭扭矩 | **同步写广播（write1_only）** | 反正要关串口了 |

**Windows 定时器精度优化：**
- Windows 默认 `time.sleep(0.001)` 实际会等 15.6ms（系统定时器精度默认 15.6ms）
- 程序开头调用 `timeBeginPeriod(1)` 把系统定时器精度提高到 1ms
- 这样动态延时才能真正精确到 1ms 粒度

---

### 2.6 数据读取（`read_data()`）— 批量读取优化

**优化说明：** 一次读 8 字节，同时拿到所有数据，采样率从 ~100 Hz 提升到 ~250 Hz。

```python
def read_data():
    # 一次读 8 字节，连续地址 56~63
    all_data, result, error = packetHandler.readTxRx(
        motor_id, SMS_STS_PRESENT_POSITION_L, 8
    )

    # 位置：低字节在前 → 有符号转换 → 转弧度
    pos_raw = all_data[0] | (all_data[1] << 8)
    pos_raw = packetHandler.scs_tohost(pos_raw, 15)
    position = pos_raw_to_rad(pos_raw)

    # 速度：低字节在前 → 有符号转换 → 转 rad/s
    speed_raw = all_data[2] | (all_data[3] << 8)
    speed_raw = packetHandler.scs_tohost(speed_raw, 15)
    speed = speed_raw * SPEED_SCALE

    # 负载/PWM占空比：低字节在前 → 有符号转换 → [-1, 1]
    load_raw = all_data[4] | (all_data[5] << 8)
    load_raw = packetHandler.scs_tohost(load_raw, 15)
    load = load_raw / 1000.0  # 1000 = 100% PWM

    # 电压：原始值 × 0.1 = V
    volts = all_data[6] * 0.1

    # 温度
    temp = float(all_data[7])

    return {"position": ..., "speed": ..., "load": load, ...}
```

**批量读取的寄存器布局（地址 56~63，连续 8 字节）：**

| 偏移 | 地址 | 内容 | 转换后单位 |
|------|------|------|-----------|
| 0-1 | 56-57 | Present Position | rad |
| 2-3 | 58-59 | Present Speed | rad/s |
| 4-5 | 60-61 | Present Load / PWM Duty | [-1, 1]（11位寄存器：bit0-9数值，bit10方向位） |
| 6 | 62 | Present Voltage | V |
| 7 | 63 | Present Temperature | °C |

**单位转换链：**
| 物理量 | 舵机原始值 | → | BAM 标准单位 |
|--------|-----------|---|-------------|
| 位置 | 计数（12 位，0~4095） | 减去中点 2048 后换算 | rad（对称坐标系） |
| 速度 | 单位（0.732 RPM） | × 0.0767 | rad/s |
| 负载/PWM | 11位：bit0-9数值，bit10方向位 | 解析后 ÷ 1000 | [-1, 1] 占空比（1000=100%） |
| 电压 | 原始值（0.1V） | × 0.1 | V |

> 位置坐标系：0 rad = 舵机中点（2048），-π rad = 一端极限，+π rad = 另一端极限。

**采样性能对比：**
| 版本 | 读取方式 | 写入方式 | 采样率 |
|------|---------|---------|--------|
| 原始版 | 4 次单独读 | 带应答 | ~100 Hz |
| 中间版 | 一次批量读 8 字节 | 普通 TxOnly | ~250 Hz |
| **最终版** | **一次批量读 8 字节** | **同步写广播（不应答）** | **~600-1000 Hz** |

---

## 三、`all_record.py` — 批量录制脚本

### 功能概述

自动化批量录制：遍历所有 P 增益 × 所有轨迹，完成完整数据集采集。

### 3.1 配置

```python
# P 增益扫描值（0-255 范围，两个舵机通用）
kps = [4, 8, 16, 32, 64]

# 激励轨迹（覆盖不同摩擦工况）
# 注：BAM 框架中没有 "brutal" 轨迹，已移除
trajectories = [
    "sin_sin",        # 多频叠加正弦（丰富频谱）
    "lift_and_drop",  # 提升后断电自由下落（Stribeck 效应）
    "up_and_down",    # 慢速升降（静摩擦、负载依赖）
    "sin_time_square" # 加速振荡（宽速度范围）
]
```

**为什么去掉 `brutal`：**
- BAM 框架的 `trajectory.py` 注册表中**不存在** `"brutal"` 这个轨迹名
- 原始代码写了但运行会直接报错 `Unknown trajectory: brutal`
- 已替换为 4 种 BAM 官方标准辨识轨迹

**BAM 可用轨迹全集（7 种）：**
| 轨迹名 | 用途 |
|--------|------|
| `sin_time_square` | sin(t²) 加速振荡，主辨识轨迹 |
| `sin_sin` | 多频叠加正弦，丰富频谱 |
| `lift_and_drop` | 断电自由下落，Stribeck 效应 |
| `up_and_down` | 慢速升降，静摩擦/负载依赖 |
| `half_sine` | 半正弦慢速运动 |
| `steps` | 阶跃信号 |
| `nothing` | 零扭矩纯重力响应 |

**总记录数：** 5 P增益 × 4 轨迹 = **20 段**（单组质量/长度配置）

### 3.2 执行流程

```python
for kp in kps:
    for trajectory in trajectories:
        # 打印当前阶段
        print(f"Kp {kp}, trajectory {trajectory}")
        
        # 调用单条记录脚本（用 subprocess.run，跨平台兼容）
        subprocess.run(command, check=True)
        
        # sin_time_square 后等待 3 秒稳定
        if trajectory == "sin_time_square":
            time.sleep(3)
```

**关键实现细节：**
- 用 `sys.executable` 确保用当前 Python 环境
- 用 `subprocess.run(check=True)` 执行，失败了直接抛错
- 跨平台兼容（Windows / Mac / Linux）

### 3.3 命令行参数

```bash
python -m bam.feetech.all_record `
    --mass 0.5 `           # 配重质量 [kg]
    --arm-mass 0.0 `       # 摆臂自身质量 [kg]（可选，默认 0.0）
    --length 0.15 `        # 摆臂长度 [m]
    --motor hd1910 `       # 电机名称（决定 P 增益地址）
    --port COM3 `          # 串口
    --id 1 `               # 舵机 ID
    --vin 7.4 `            # 供电电压（默认 7.4V）
    --logdir data_raw      # 输出目录
```

---

## 四、`actuators.py` — 全局注册表

BAM 框架通过 `bam/actuators.py` 统一注册所有支持的执行器：

```python
from .feetech.actuator import STS3215Actuator, HD1910Actuator

actuators = {
    # ... 其他舵机 ...
    # Feetech STS3215
    "sts3215": lambda: STS3215Actuator(Pendulum),
    # Feetech HD1910
    "hd1910": lambda: HD1910Actuator(Pendulum),
    # ...
}
```

**区分方式：**
| 阶段 | 参数 | 作用 | 可选值 |
|------|------|------|--------|
| 硬件录制 | `--motor` | ① 选择 P 增益寄存器地址 ② 数据标签（存 JSON） | `sts3215` / `hd1910` |
| 拟合/仿真 | `--actuator` | 加载对应仿真模型 | `sts3215` / `hd1910` |

> 硬件录制时必须指定正确的 `--motor`，因为两个型号的 P 增益寄存器地址不同（21 vs 50）。

---

## 五、与 BAM 框架的集成关系

```
                    ┌─────────────────────────┐
                    │   bam/feetech/         │
                    │  ┌───────────────────┐  │
                    │  │ STS3215Actuator   │  │ ← 仿真/拟合时使用
                    │  │ HD1910Actuator   │  │
                    │  └───────────────────┘  │
                    │  ┌───────────────────┐  │
                    │  │ record.py         │  │ ← 硬件采集（通用）
                    │  │ all_record.py     │  │
                    │  └───────────────────┘  │
                    └─────────┬───────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        bam/actuators.py   bam/fit.py    bam/mujoco.py
        (注册表)          (参数拟合)     (仿真集成)
```

**仿真路径：**
`STS3215Actuator` / `HD1910Actuator` → `bam/fit.py` 拟合参数 → `bam/mujoco.py` 集成到 MuJoCo

**硬件路径：**
`record.py` → 连接真实舵机 → 数据 → `bam/process.py` 预处理 → `bam/fit.py` 拟合

---

## 六、修改前后对比

| 项目 | 修改前 | 修改后 |
|------|--------|--------|
| **通信库** | pypot.feetech.FeetechSTS3215IO | scservo_sdk（官方 ftservo-python-sdk） |
| **速率限制** | 有（q_target_smooth + max_velocity） | 无（直接跟踪目标） |
| **error_gain** | 0.166 | **0.163**（实测标定：KP=1 时 4000 计数 → 100% PWM） |
| **位置精度** | 15 位假设 | 12 位实际（4096 计数/圈） |
| **速度单位** | 0.1°/s | 0.732 RPM/单位 |
| **P 增益地址** | 未明确 | 按型号自动选择（STS3215=21, HD1910=50） |
| **默认电压** | 15.0V / 5.1V | 7.4V（STS3215/HD1910 标称） |
| **stateful** | True（有内部状态） | 无状态（纯函数） |
| **轨迹列表** | 5 条（含无效的 brutal） | 4 条（官方标准轨迹） |
| **支持舵机** | 仅 STS3215 | STS3215 + **HD1910** |
| **读取方式** | 4 次单独读 | **一次批量读 8 字节**（采样率 ~100Hz → ~250Hz） |
| **录制写入** | 带应答（write2） | **同步写广播（write2_only）**（舵机不应答，无残留） |
| **扭矩切换** | 带应答（write1） | **同步写广播（write1_only）** |
| **延时策略** | 固定 2ms | **动态延时补偿**（每帧至少 1ms）+ Windows timeBeginPeriod(1) |
| **预热机制** | 1秒每帧都写 | **最多重试10次，成功后等1秒稳定** |
| **回零逻辑** | 直接用最后一帧位置 | **先等0.5s稳定，再读当前实际位置作为起点** |
| **错误处理** | 无 | **try-finally 安全退出**（关扭矩+关串口） |
| **批量执行** | os.system（Windows 不兼容） | **subprocess.run**（跨平台） |
| **语音播报** | 有（Windows 有 bug） | 已删除 |
| **位置坐标系** | 0 rad = 舵机一端（0 计数） | **0 rad = 中点（2048 计数）**（对称坐标系） |
| **P 增益扫描数** | 4 个值 | **5 个值**（4, 8, 16, 32, 64） |
| **摆臂质量参数** | 无 | **--arm-mass**（可选，默认 0.0） |
| **PWM 解析** | 16位有符号（bit15符号位） | **11位：bit0-9数值，bit10方向位** |
| **采样率** | ~100 Hz | **~500 Hz**（批量读 + 同步写广播） |



