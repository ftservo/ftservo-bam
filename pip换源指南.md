# pip 换源指南（国内加速）

> 国内访问 PyPI 官方源速度慢、不稳定，换成国内镜像源可以大幅提升下载速度。

---

## 一、推荐镜像源

| 源 | 地址 | 优先级 | 说明 |
|---|------|--------|------|
| **华为云源** | https://repo.huaweicloud.com/repository/pypi/simple | ⭐⭐⭐⭐⭐ | 推荐首选，稳定快速 |
| 中科大源 | https://pypi.mirrors.ustc.edu.cn/simple | ⭐⭐⭐⭐ | 常用，稳定 |
| 清华源 | https://pypi.tuna.tsinghua.edu.cn/simple | ⭐⭐⭐⭐ | 常用，稳定 |
| 阿里源 | https://mirrors.aliyun.com/pypi/simple | ⭐⭐⭐⭐ | 常用，速度快 |
| 豆瓣源 | https://pypi.douban.com/simple | ⭐⭐⭐ | 备用 |
| 官方源 | https://pypi.org/simple | ⭐ | 国外，慢 |

---

## 二、临时换源（单次使用）

在安装命令后面加 `-i` 参数，只对本次生效：

```powershell
# 使用华为云源（推荐）
pip install 包名 -i https://repo.huaweicloud.com/repository/pypi/simple

# 示例：安装 mujoco
pip install mujoco -i https://repo.huaweicloud.com/repository/pypi/simple
```

**适用场景：** 偶尔安装一个包，不想改全局配置。

---

## 三、永久换源（推荐）

### 方法一：命令行一键配置（Windows PowerShell）

```powershell
# 配置华为云源为默认
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple

# 验证是否配置成功
pip config list
```

输出应该显示：
```
global.index-url='https://repo.huaweicloud.com/repository/pypi/simple'
```

### 方法二：手动修改配置文件

**Windows 配置文件位置：**
```
C:\Users\你的用户名\AppData\Roaming\pip\pip.ini
```

如果 `pip` 文件夹不存在，手动创建。

**pip.ini 文件内容：**

```ini
[global]
index-url = https://repo.huaweicloud.com/repository/pypi/simple
trusted-host = repo.huaweicloud.com

[install]
trusted-host = repo.huaweicloud.com
```

**说明：**
- `index-url`：默认下载源地址
- `trusted-host`：信任该域名，避免 SSL 警告

---

## 四、常用源切换命令

```powershell
# 华为云源（推荐）
pip config set global.index-url https://repo.huaweicloud.com/repository/pypi/simple

# 中科大源
pip config set global.index-url https://pypi.mirrors.ustc.edu.cn/simple

# 清华源
pip config set global.index-url https://pypi.tuna.tsinghua.edu.cn/simple

# 阿里源
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple

# 恢复官方源
pip config unset global.index-url
```

---

## 五、虚拟环境换源说明

- 虚拟环境会**继承**全局 pip 配置
- 只要全局配置了换源，虚拟环境里直接用就行，不用重复配置
- 如果想单独给虚拟环境换源，进入虚拟环境后执行同样的 `pip config set` 命令即可

---

## 六、常见问题

### Q1: 换源后还是慢？
A: 检查是不是在虚拟环境里，虚拟环境可能有自己的配置。用 `pip config list` 看看当前生效的配置。

### Q2: 安装时报 SSL 错误？
A: 在配置里加上 `trusted-host` 那一行，或者临时安装时加 `--trusted-host` 参数：
```powershell
pip install 包名 -i https://repo.huaweicloud.com/repository/pypi/simple --trusted-host repo.huaweicloud.com
```

### Q3: 怎么恢复官方源？
A: 执行：
```powershell
pip config unset global.index-url
```

### Q4: 华为云源没有某个包怎么办？
A: 切换到中科大源、清华源或阿里源试试，或者临时用官方源安装：
```powershell
pip install 包名 -i https://pypi.org/simple
```

---

