# BAM 项目操作步骤指南

> **BAM (Better Actuator Models)** — 伺服执行器摩擦模型辨识与仿真工具库
> 仓库：https://github.com/Rhoban/bam
> 论文：Extended Friction Models for the Physics Simulation of Servo Actuators (ICRA 2025)

---

## 一、项目概述

BAM 是一个 Python 库，用于**辨识**和**使用**伺服执行器（舵机）的高级摩擦模型。它比 MuJoCo 等仿真器默认的库仑-粘性摩擦模型更精确，能捕捉 Stribeck 效应、负载依赖性、方向性摩擦等复杂现象。

**核心功能：**
- 预辨识模型库：Dynamixel MX-64/106、XL-320/330、Feetech STS3215、Waveshare ST3025、eRob80:50/100
- MuJoCo CPU 集成 API
- MuJoCo Warp (GPU / mjlab) 集成 API
- 完整的参数辨识流水线（硬件搭建 → 数据采集 → 参数拟合）

---

## 二、环境准备与安装

### 2.1 系统要求

- **Python 版本：** 3.12.x（注意：不支持 3.13+）
- **操作系统：** Linux / macOS / Windows
- **推荐工具：** uv（Python 包管理工具，比 pip 快 10-100 倍）

> **uv 安装（Windows PowerShell）：**
> ```powershell
> powershell -c "irm https://astral.sh/uv/install.ps1 | iex"
> ```
> 安装后重启终端，运行 `uv --version` 确认安装成功。

### 2.2 安装方式

> ⚠️ **如果是修改后的源码，必须用源码安装方式，不能用 PyPI 包安装。**
> PyPI 包 `better-actuator-models` 是官方发布的版本，不包含你本地修改的代码。

> 💡 **Windows PowerShell 续行说明：**
> Windows PowerShell 里多行命令的续行符是 **反引号 `` ` ``**（键盘左上角 ESC 下面那个键），**不是反斜杠 `\`**。
> 下面所有命令示例均已按 PowerShell 语法编写，直接复制即可。

#### 方式一：源码安装（推荐，修改代码 / 辨识新电机）

> 虚拟环境的目的：**测试修改代码后的程序**，不污染系统 Python。
> 虚拟环境与全局 Python 互不共享，所有依赖都要在虚拟环境里单独装一遍。

**方法 A：uv run 自动创建（一条命令搞定）**

```powershell
# 1. 进入项目目录
cd D:\gitree\bam

# 2. 直接运行命令，uv run 自动创建虚拟环境并安装依赖
#    --extra identification 安装辨识工具链（ftservo-python-sdk、optuna、cmaes 等）
#    第一次运行会自动创建 .venv 并安装所有依赖，之后直接跑就行
uv run --extra identification python -m bam.feetech.record --help
```

**方法 B：先 sync 再 run（两步）**

```powershell
# 1. 进入项目目录
cd D:\gitree\bam

# 2. 创建虚拟环境并同步所有依赖
uv venv .venv
uv sync --extra identification

# 3. 之后直接 run
uv run python -m bam.feetech.record --help
```

**说明：**
- 两种方法效果一样，都是创建 `.venv` 虚拟环境并安装依赖
- 可编辑模式安装，改代码立即生效，不用重装
- 用完删 `.venv` 文件夹就彻底卸载，无残留

> **依赖包说明：**
> - `--extra identification`：只装**辨识工具链**（ftservo-python-sdk、optuna、cmaes、matplotlib 等），用于数据采集和参数拟合
> - `--extra mujoco`：额外装 MuJoCo 物理仿真引擎，用于场景二的仿真
> - `--extra all`：全部都装（辨识 + MuJoCo + mjlab GPU）
>
> 如果只做辨识（场景一），用 `--extra identification` 就够了；
> 如果还要做 MuJoCo 仿真（场景二），改成 `--extra identification,mujoco` 或直接 `--extra all`。

---

## 三、场景一：辨识 Feetech 舵机（完整流水线）

以 HD1910 为例，完整走一遍辨识流程。STS3215 操作完全相同，只需把 `--motor hd1910` 改成 `--motor sts3215`。

### 3.1 硬件搭建：摆锤测试台

#### 所需硬件

1. **刚性摆臂（多种长度）**
   - 质量远小于末端负载（3D 打印或激光切割木质）
   - 大功率电机使用金属臂

2. **配重块（多种质量）**
   - 重量足够大，覆盖宽范围负载扭矩

3. **执行器安装支架**
   - 确保舵机在快速运动时牢固固定
   - 摆臂能在垂直方向 ±90° 范围内摆动

4. **通信接口**
   - Feetech: FT-URT2 USB 转串口通信板
   - 其他舵机：对应品牌的 USB 通信板

#### 关键参数记录

搭建完成后必须测量并记录：
- 配重块质量 `mass` [kg]
- 摆臂长度 `length` [m]
- 摆臂自身质量 `arm_mass` [kg]

### 3.2 执行器模型（Feetech 已内置）

Feetech STS3215 / HD1910 的执行器模型已经内置在 `bam/feetech/actuator.py` 中，直接使用即可，不需要自己写代码。

#### 内置支持的 Feetech 舵机

| 型号 | 标识 | 说明 |
|------|------|------|
| STS3215 | `sts3215` | 金属齿轮数字舵机 |
| HD1910 | `hd1910` | 金属齿轮数字舵机 |

#### 模型初始参数（HD1910）

| 参数 | 初始值 | 说明 |
|------|--------|------|
| 额定电压 | 7.4 V | 供电电压 |
| P 增益 | 32 | 固件默认值 |
| error_gain | 0.163 | 标定：KP=1 时 4000 计数误差 → 100% PWM |
| 扭矩常数 $k_t$ | 0.64 Nm/A | 空载速度法估算 |
| 电机电阻 $R$ | 5.0 Ω | 经验估计值 |
| 等效惯量 $I$ | 0.00005 kg·m² | 经验估计值 |
| 位置精度 | 12 位（4096 计数/圈） | 磁编码 |
| 速度单位 | 0.732 RPM/单位 | 固件内置单位 |

> 详细源码解读见 `feetech源码解读_修改后完整版.md`

### 3.3 数据采集（Feetech HD1910 示例）

#### 四种激励轨迹

每种轨迹运行 6 秒，覆盖不同摩擦工况：

| 轨迹名称 | 描述 | 激励的摩擦效应 |
|---------|------|--------------|
| `sin_time_square` | $\sin(t^2)$ 加速振荡 | 宽速度范围，通用 |
| `lift_and_drop` | 提升到 -90° 后断电自由下落 | 分离粘性摩擦与反电动势阻尼 |
| `up_and_down` | 0 → +90° → 0.8·90° 慢速运动 | 静摩擦、负载依赖性 |
| `sin_sin` | 多频叠加正弦 | 丰富多频内容 |

#### 单条轨迹录制（HD1910）

```powershell
uv run python -m bam.feetech.record `
    --port COM3 `
    --id 1 `
    --motor hd1910 `
    --mass 0.5 `
    --arm-mass 0.05 `
    --length 0.15 `
    --vin 7.4 `
    --kp 32 `
    --trajectory sin_time_square `
    --logdir data_raw_hd1910
```

**命令参数说明：**

| 参数 | 示例值 | 说明 |
|------|--------|------|
| `--port` | `COM3` | 串口通信板设备名（Windows 是 COMx，Linux 是 /dev/ttyUSBx） |
| `--id` | `1` | 舵机 ID（出厂默认 1，可通过舵机设置软件修改） |
| `--motor` | `hd1910` | 电机型号标识：① 决定 P 增益寄存器地址 ② 写入数据文件的标签 |
| `--mass` | `0.5` | 末端配重块质量 [kg]，必须和实际称重一致 |
| `--arm-mass` | `0.05` | 摆臂自身质量 [kg]（可选，默认 0.0） |
| `--length` | `0.15` | 摆臂长度 [m]，从舵机轴心到配重中心的距离 |
| `--vin` | `7.4` | 实际供电电压 [V]，默认 7.4V |
| `--kp` | `32` | 固件 P 增益（范围 0-255，默认 32） |
| `--trajectory` | `sin_time_square` | 激励轨迹名称（见上面的轨迹表） |
| `--logdir` | `data_raw_hd1910` | 数据输出目录（相对路径，相对于当前工作目录） |

> 数据保存位置：`{当前目录}\{logdir}\{日期}.json`，例如 `D:\gitree\bam\data_raw_hd1910\2026-09-18.json`

#### 单条轨迹录制（STS3215）

```powershell
uv run python -m bam.feetech.record `
    --port COM3 `
    --id 1 `
    --motor sts3215 `
    --mass 0.5 `
    --arm-mass 0.05 `
    --length 0.15 `
    --vin 7.4 `
    --kp 32 `
    --trajectory sin_time_square `
    --logdir data_raw_sts3215
```

#### 批量录制（自动遍历所有 P 增益和轨迹）

```powershell
uv run python -m bam.feetech.all_record `
    --port COM3 `
    --id 1 `
    --motor hd1910 `
    --mass 0.5 `
    --arm-mass 0.05 `
    --length 0.15 `
    --vin 7.4 `
    --logdir data_raw_hd1910
```

> 注：`--motor` 不仅是数据标签，还会影响 P 增益寄存器地址的选择：
> - `sts3215` → 写地址 21
> - `hd1910` → 写地址 50
>
> 其他寄存器（位置、速度、电压、温度）两个型号地址相同。

#### 录制策略

1. 对**每种质量/长度组合**重复运行录制命令
2. P 增益值：`[4, 8, 16, 32, 64]`（HD1910 默认范围）
3. 5 个 P 增益 × 4 条轨迹 = 20 段记录（单个配重组合）

**建议：** 至少测试 3 种不同配重（如 0.2kg / 0.5kg / 1.0kg），覆盖宽负载范围。

#### 数据格式

每段记录生成一个 JSON 文件：

```json
{
  "mass": 0.5,
  "arm-mass": 0.05,
  "length": 0.15,
  "kp": 32,
  "vin": 7.4,
  "motor": "hd1910",
  "trajectory": "sin_time_square",
  "entries": [
    {
      "timestamp": 0.004,
      "position": 0.0015,
      "speed": 0.024,
      "load": 0.05,
      "input_volts": 7.4,
      "temp": 35.0,
      "goal_position": 0.0,
      "torque_enable": true
    }
  ]
}
```

**字段说明：**
| 字段 | 单位 | 说明 |
|------|------|------|
| `mass` | kg | 末端配重质量 |
| `arm-mass` | kg | 摆臂自身质量（默认 0.0） |
| `length` | m | 摆臂长度 |
| `kp` | — | 固件 P 增益 |
| `vin` | V | 供电电压 |
| `motor` | string | 电机型号标识 |
| `trajectory` | string | 轨迹名称 |
| `position` | rad | 实际位置 |
| `speed` | rad/s | 实际速度 |
| `load` | [-1, 1] | **PWM 占空比反馈**（1.0 = 100% PWM） |
| `input_volts` | V | 供电电压 |
| `temp` | °C | 舵机温度 |
| `goal_position` | rad | 目标位置 |
| `torque_enable` | bool | 扭矩是否开启 |

> **采样率：** ~250 Hz（批量读取优化后），每轮循环读 8 字节一次拿全部数据。

### 3.4 数据预处理（Feetech HD1910）

#### 检查抖动

```powershell
uv run python -m bam.jitter --logdir data_raw_hd1910/
```

#### 重采样到固定时间步

```powershell
uv run python -m bam.process `
    --raw data_raw_hd1910 `
    --logdir data_processed_hd1910 `
    --dt 0.005
```

`--dt` 为目标时间步长（秒），默认 5ms。

#### 绘制原始数据

```powershell
uv run python -m bam.plot `
    --actuator hd1910 `
    --logdir data_processed_hd1910
```

### 3.5 参数拟合（Feetech HD1910）

#### 运行拟合（M6 模型）

```powershell
uv run python -m bam.fit `
    --actuator hd1910 `
    --model m6 `
    --logdir data_processed_hd1910 `
    --output params/hd1910/m6.json
```

#### STS3215 拟合命令

```powershell
uv run python -m bam.fit `
    --actuator sts3215 `
    --model m6 `
    --logdir data_processed_sts3215 `
    --output params/sts3215/m6.json
```

#### 关键参数说明

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--method` | `cmaes` | 优化算法：cmaes / random / nsgaii |
| `--trials` | 100000 | 评估次数（复杂模型 M5/M6 建议增加） |
| `--workers` | 1 | 并行工作进程数 |
| `--validation_kp` | — | 留出某个 P 增益值作为验证集（如 8） |
| `--wandb` | — | 启用 Weights & Biases 日志 |
| `--load-study` | — | 从已有 Optuna study 断点续跑 |

#### 建议：拟合所有 6 个模型并比较

```powershell
$models = @("m1", "m2", "m3", "m4", "m5", "m6")
foreach ($model in $models) {
    uv run python -m bam.fit `
        --actuator hd1910 `
        --model $model `
        --logdir data_processed_hd1910 `
        --validation_kp 8 `
        --output params/hd1910/$model.json `
        --trials 100000
}
```

**选择标准：** 验证 MAE 最低且参数数量可接受的模型。

#### 评估与可视化

```powershell
# 评估参数文件
uv run python -m bam.fit `
    --actuator hd1910 `
    --model m6 `
    --logdir data_processed_hd1910 `
    --eval `
    --output params/hd1910/m6.json

# 绘制仿真 vs 实测对比图
uv run python -m bam.plot `
    --actuator hd1910 `
    --logdir data_processed_hd1910 `
    --sim `
    --params params/hd1910/m6.json
```

支持同时传入多个 `--params` 对比不同模型。

---

## 四、场景二：使用预辨识模型（MuJoCo 集成）

如果你的舵机已经在 BAM 模型库中，或者你已经完成了辨识得到了参数文件，就可以直接加载参数用于仿真。

### 4.1 支持的预辨识舵机

| 名称 | 标识 | 品牌型号 |
|------|------|---------|
| MX-64 | `mx64` | ROBOTIS Dynamixel MX-64 |
| MX-106 | `mx106` | ROBOTIS Dynamixel MX-106 |
| XL-320 | `xl320` | ROBOTIS Dynamixel XL-320 |
| XL-330 | `xl330` | ROBOTIS Dynamixel XL-330 |
| STS3215 (7.4V) | `feetech_sts3215_7_4V` | Feetech STS3215 |
| HD1910 | `feetech_hd1910` | Feetech HD1910 |
| ST3025 | `waveshare_st3025` | Waveshare ST3025 总线舵机 |
| eRob80:50 | `erob80_50` | ZeroErr eRob80:50 |
| eRob80:100 | `erob80_100` | ZeroErr eRob80:100 |

### 4.2 摩擦模型变体

从 M1 到 M6，复杂度递增：

| 模型 | 参数数量 | 说明 |
|------|---------|------|
| M1 | 标准库仑-粘性摩擦 | MuJoCo 默认模型 |
| M2 | + Stribeck 效应 | 5 参数 |
| M3 | + 负载依赖摩擦 | 3 参数增量 |
| M4 | + Stribeck 负载依赖 | 7 参数 |
| M5 | + 方向性摩擦 | 9 参数 |
| M6 | + 二次负载摩擦 | 11 参数 |

### 4.3 MuJoCo CPU 集成步骤

#### 步骤 1：加载模型参数

```python
from bam.model import load_model

# 加载内置预辨识模型
model = load_model(motor_name="mx64", model="m6")

# 加载 Feetech HD1910 模型
model = load_model(motor_name="feetech_hd1910", model="m6")

# 或者加载自定义 JSON 参数文件（自己辨识得到的）
# model = load_model("path/to/your_params.json")
```

#### 步骤 2：配置 MuJoCo XML

在 MJCF 中，每个执行器必须声明为 `motor` 类型（不能是 `position` 或 `velocity`）：

```xml
<actuator>
  <motor name="joint_1" joint="joint_1" gear="1"/>
  <motor name="joint_2" joint="joint_2" gear="1"/>
</actuator>
```

> 注意：BAM 会在运行时覆盖 `frictionloss`、`damping` 和 `armature`，XML 中设置的这些值会被忽略。

#### 步骤 3：创建控制器

```python
import mujoco
from bam.mujoco import MujocoController

# 加载机器人模型
mj_model = mujoco.MjModel.from_xml_file("robot.xml")
mj_data = mujoco.MjData(mj_model)

# 创建 BAM 控制器
controller = MujocoController(
    model=model,                          # 加载的摩擦模型
    actuator=["joint_1", "joint_2"],      # 关节名称列表（对应 XML 中的 motor name）
    mujoco_model=mj_model,
    mujoco_data=mj_data,
    # 可选：电压降模型（模拟电池内阻）
    vin_drop_resistance=0.1,              # 0.1 欧姆内阻
    vin_min=6.0,                          # 最低电压 [V]
)
```

#### 步骤 4：仿真循环

```python
mujoco.mj_resetData(mj_model, mj_data)

joint_names = ["joint_1", "joint_2"]
target_angles = [0.5, -0.3]  # 目标关节角 [rad]

while True:
    # 设置每个关节的目标位置
    for name, angle in zip(joint_names, target_angles):
        controller.set_q_target(name, angle)
    
    # 更新 BAM 控制器（计算摩擦、扭矩等）
    controller.update()
    
    # MuJoCo 步进
    mujoco.mj_step(mj_model, mj_data)
```

### 4.4 多执行器配置（机器人多关节）

对于多关节机器人，使用 JSON 配置文件批量加载：

```python
from bam.mujoco import load_config

controllers, dof_to_controller = load_config(
    path="config.json",
    mujoco_model=mj_model,
    mujoco_data=mj_data,
    kp=125.0,        # 比例增益
    vin=7.5,         # 供电电压
)
```

配置文件结构示例：

```json
{
  "arm": {
    "dofs": ["shoulder", "elbow"],
    "model": {
      "actuator": "mx64",
      "model": "m6",
      "kt": 1.62,
      "R": 3.95,
      "armature": 0.012,
      "friction_base": 0.09,
      "friction_viscous": 0.012
    },
    "error_gain": 1.0,
    "max_pwm": 885
  }
}
```

---

## 五、模型参数文件说明

拟合完成后生成的 JSON 参数文件结构：

```json
{
  "model": "m6",
  "actuator": "your_motor",
  "kt": 2.21,
  "R": 2.03,
  "armature": 0.026,
  "friction_base": 1.0e-05,
  "friction_viscous": 0.051,
  "friction_stribeck": 0.122,
  "dtheta_stribeck": 1.75,
  "alpha": 1.14,
  "load_friction_motor": 0.198,
  "load_friction_external": 0.022,
  "load_friction_motor_stribeck": 0.199,
  "load_friction_external_stribeck": 0.087,
  "load_friction_motor_quad": 0.010,
  "load_friction_external_quad": 7.3e-05,
  "q_offset": 0.0
}
```

| 参数 | 说明 |
|------|------|
| `kt` | 扭矩常数 [Nm/A] |
| `R` | 电机电阻 [Ω] |
| `armature` | 等效转动惯量 [kg·m²] |
| `friction_base` | 库仑摩擦基础值 |
| `friction_viscous` | 粘性摩擦系数 |
| `friction_stribeck` | Stribeck 效应摩擦 |
| `dtheta_stribeck` | Stribeck 特征速度 |
| `alpha` | Stribeck 曲线形状指数 |
| `load_friction_*` | 负载相关摩擦（电机侧/外部侧） |
| `q_offset` | 安装位置偏移 [rad] |

---

## 六、完整操作流程总结

### 快速上手（使用预辨识模型）

```
安装 → load_model() → MujocoController → 仿真循环
```

### 辨识新舵机（完整流水线）

```
硬件搭建 → 执行器建模 → 数据采集 → 数据预处理 → 参数拟合 → 验证评估 → 集成仿真
   ↓           ↓           ↓           ↓           ↓           ↓
摆锤测试台  继承基类    all_record  process     fit         plot对比
```

---

## 七、常见问题

### Q1: 支持哪些舵机？
A: 内置支持 Dynamixel (MX/XL 系列)、Feetech STS3215、Waveshare ST3025、eRob80。其他品牌可通过继承基类自行扩展。

### Q2: 需要什么硬件？
A: 摆臂 + 配重块 + 舵机安装支架 + 通信线。最简单的方式是用 3D 打印摆臂 + 标准砝码。

### Q3: 拟合需要多久？
A: 取决于模型复杂度和 trial 数量。M1-M2 较快，M5-M6（复杂模型）可能需要数小时。建议多 worker 并行。

### Q4: 选哪个模型最好？
A: 不一定是 M6 最好。建议拟合全部 6 个模型，选验证 MAE 最低的那个。简单舵机可能 M3/M4 就够了。

### Q5: 如何贡献新模型？
A: 辨识完成后，将参数 JSON 和原始数据提交到 BAM 仓库的 `bam/params/` 目录，并在文档中添加说明。

---

## 八、参考资源

- **官方文档：** https://bam.readthedocs.io/
- **论文原文：** https://arxiv.org/abs/2410.08650
- **原始数据：** HuggingFace - buckots/Gregwar/bam_data
- **示例项目：** Open Duck Mini（使用 BAM 标定舵机的开源人形机器人）
