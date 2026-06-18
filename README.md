# Domain Rand — MuJoCo 域随机化数据集生成工具

基于 MuJoCo 的 Domain Randomization 数据集制作工具，支持：

- **离线 DR 数据集**：视觉域随机化（纹理/颜色、光照、相机），静态场景批量渲染
- **交互式 IL 示教采集**：Task + Policy 解耦架构，键盘遥操作录制多帧轨迹

## 安装

```bash
# 创建 conda 环境
conda create -n domain_rand python=3.11 -y
conda activate domain_rand

# 安装依赖
pip install -r requirements.txt
```

## 快速开始

```bash
# 使用默认配置生成 100 个 episode 的数据集
python scripts/generate_dataset.py \
    --config configs/default.yaml \
    --num-episodes 100 \
    --output ./datasets/my_dataset.h5

# 使用重度视觉 DR 预设
python scripts/generate_dataset.py \
    --config configs/presets/heavy_visual_dr.yaml \
    --num-episodes 500

# 可视化生成的数据集
python scripts/visualize_dataset.py --dataset ./datasets/my_dataset.h5
```

## 项目结构

```
domain_rand/
├── pyproject.toml              # 项目配置与依赖
├── configs/                    # YAML 配置文件
│   ├── default.yaml            # 默认随机化参数
│   └── presets/                # 预设方案
│       ├── light_dr.yaml       # 仅光照 DR
│       ├── heavy_visual_dr.yaml
│       └── full_dr.yaml
├── src/domain_rand/
│   ├── core/                   # 核心组件
│   │   ├── config.py           # 配置系统（dataclass + YAML）
│   │   ├── distributions.py    # 随机分布（Uniform/LogUniform/Normal/Choice）
│   │   └── scene.py            # 场景管理 + 动力学辅助方法
│   ├── randomizers/            # 随机化器
│   │   ├── base.py             # 抽象基类 + 组合器
│   │   ├── texture.py          # 纹理/颜色随机化
│   │   ├── lighting.py         # 光照随机化
│   │   ├── camera.py           # 相机随机化
│   │   └── placement.py        # 物体放置随机化
│   ├── tasks/                  # 🆕 Task 抽象（用户实现接口）
│   │   └── base.py             # Task ABC
│   ├── policy/                 # 🆕 策略抽象（动作生成）
│   │   ├── base.py             # Policy ABC
│   │   └── keyboard_teleop.py  # 键盘遥操作
│   ├── pipeline/               # 数据采集管线
│   │   ├── collector.py        # 离线 DR 采集循环
│   │   ├── demo_collector.py   # 🆕 交互式 IL 采集循环
│   │   ├── recorder.py         # 单帧 HDF5 记录器
│   │   ├── il_recorder.py      # 🆕 向后兼容重导出（→ recorders.simple）
│   │   ├── recorders/          # 🆕 可插拔 HDF5 记录器
│   │   │   ├── base.py         #   BaseRecorder ABC
│   │   │   ├── simple.py       #   SimpleRecorder（灵活 dict，默认）
│   │   │   └── robomimic.py    #   RobomimicRecorder
│   │   └── metadata.py         # 元数据管理
│   └── utils/
│       └── rendering.py        # RGB/深度渲染 + 相机矩阵
├── assets/scenes/              # MuJoCo 场景文件
│   ├── example_factory.xml     # 示例场景（箱子+物体）
│   └── table_with_stl.xml      # 桌子+STL物体场景
├── scripts/                    # 运行脚本
│   ├── generate_dataset.py     # 离线数据集生成入口
│   ├── collect_demos.py        # 🆕 交互式 IL 示教采集入口
│   ├── visualize_dataset.py    # 数据集可视化
│   └── verify_dataset.py       # 数据集结构验证
└── tests/                      # 单元测试
```

## 数据集格式（HDF5）

```
dataset.h5
├── .attrs/                  # 全局元数据
│   ├── config_json          # 完整配置（JSON）
│   ├── num_episodes
│   ├── resolution           # [H, W]
│   ├── save_depth
│   ├── created_at           # ISO 8601 时间戳
│   └── version
├── episode_0000/
│   ├── rgb                  # (H, W, 3) uint8
│   ├── depth                # (H, W) float32
│   ├── extrinsics           # (4, 4) float32
│   ├── intrinsics           # (3, 3) float32
│   └── .attrs/
│       └── meta             # JSON 元数据（相机名、随机化参数等）
├── episode_0001/
│   └── ...
└── ...
```

## 配置说明

### 纹理随机化 (`texture`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用 |
| `mode` | `"rgb"` \| `"checker"` \| `"noise"` \| `"gradient"` | `"rgb"` | 随机化模式 |
| `rgba_range` | [8 floats] | `[0.2,1.0, 0.2,1.0, 0.2,1.0, 1.0,1.0]` | RGBA 各通道范围 |
| `exclude_geoms` | list\[str\] | `[]` | 排除的 geom 名称子串 |

### 光照随机化 (`lighting`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用 |
| `position_jitter` | float | `0.5` | 光源位置抖动范围（米） |
| `diffuse_range` | [2 floats] | `[0.3, 1.0]` | 漫反射颜色范围 |
| `ambient_range` | [2 floats] | `[0.1, 0.5]` | 环境光颜色范围 |
| `specular_range` | [2 floats] | `[0.0, 0.5]` | 镜面反射颜色范围 |
| `direction_jitter` | float | `0.3` | 方向抖动（弧度） |
| `random_toggle` | bool | `false` | 随机开关光源 |

### 相机随机化 (`camera`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `enabled` | bool | `true` | 是否启用 |
| `position_jitter` | [3 floats] | `[0.1, 0.1, 0.05]` | 各轴抖动范围（米） |
| `rotation_jitter` | float | `0.05` | 四元数抖动 |
| `fovy_range` | [2 floats] | `[40.0, 70.0]` | 视场角范围（度） |
| `random_camera` | bool | `true` | 多相机时随机选择 |

### 数据集 (`dataset`)

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `num_episodes` | int | `1000` | 总 episode 数 |
| `frames_per_episode` | int | `1` | 每 episode 帧数 |
| `render_height` | int | `480` | 渲染高度 |
| `render_width` | int | `640` | 渲染宽度 |
| `save_depth` | bool | `true` | 是否保存深度图 |

## IL 示教采集（交互式）

除了离线静态渲染，项目还提供了**交互式示教采集管线**，用于为 Imitation Learning 录制多帧轨迹数据（如 Push-T、Pick-and-Place 等机器人任务）。

### 架构

两条管线并存，互不干扰：

| 管线 | 入口 | 循环 |
|------|------|------|
| **离线 DR** | `DatasetCollector` + `generate_dataset.py` | `randomize` → `mj_forward` → render ×1 → record |
| **交互式 IL** | `DemoCollector` + `collect_demos.py` | `randomize` → `task.reset` → loop { `policy.get_action` → `task.step` → render } → record |

核心抽象：

```
┌──────────────────────┐     ┌──────────────────────────┐
│ Task (你实现)        │     │ Policy (你选择/实现)      │
│                      │     │                          │
│ reset(scene, rng)    │     │ get_action(obs) → action │
│ step(scene, action)  │     │                          │
│ get_observation()    │     │ 内置: KeyboardTeleop     │
└──────┬───────────────┘     └──────────┬───────────────┘
       │ 注入                           │ 注入
       └──────────┬─────────────────────┘
                  ▼
     ┌────────────────────────┐
     │ DemoCollector          │  ← 框架提供，task 作者不需要碰
     │  • 场景加载 + DR       │
     │  • 渲染 + OpenCV 显示  │
     │  • HDF5 轨迹记录       │
     │  • episode 管理        │
     └────────────────────────┘
```

- **`Task`**（[tasks/base.py](src/domain_rand/tasks/base.py)）— 你写新 task 时**唯一需要实现的接口**
- **`Policy`**（[policy/base.py](src/domain_rand/policy/base.py)）— 动作生成器，内置 `KeyboardTeleop`
- **`DemoCollector`**（[pipeline/demo_collector.py](src/domain_rand/pipeline/demo_collector.py)）— 注入 Task + Policy，管理完整采集循环
- **`ILRecorder`**（[pipeline/il_recorder.py](src/domain_rand/pipeline/il_recorder.py)）— 多帧 HDF5 轨迹记录器

### IL 数据集格式（HDF5）

> 以下为默认 `SimpleRecorder` 的输出格式。可通过 `il_demo.recorder` 切换为 [robomimic 格式](#格式-2robomimicrecorder) 或[自定义格式](#格式-3自定义-recorder)。

```
demos.h5
├── .attrs/
│   ├── config_json          # 完整配置
│   ├── num_episodes
│   ├── action_dim           # action 向量维度
│   ├── state_dim            # state 向量维度
│   └── task_type / policy_type
├── episode_0000/
│   ├── observations/
│   │   ├── rgb              # (T, H, W, 3) uint8  ← 帧序列
│   │   ├── state            # (T, D) float32      ← 状态序列
│   │   └── depth            # (T, H, W) float32  [可选]
│   ├── actions              # (T, A) float32
│   ├── rewards              # (T,) float32
│   ├── dones                # (T,) bool
│   └── .attrs/meta          # episode 元数据
└── ...
```

### 写一个新 Task（开发指引）

实现一个自定义 task 只需 **3 步**：

---

#### 第 1 步：创建 MuJoCo 场景 XML

在 `assets/scenes/` 下创建场景文件，包含你的机器人、操作物体、目标标记和相机。

关键约定 —— 相机挂载方式：

```xml
<!-- 世界固定相机（参考视角） -->
<camera name="cam_overhead" pos="0 1.5 2.0" quat="..." fovy="50" />

<!-- 眼在手上相机：挂在机器人末端 body 下，跟随运动 -->
<body name="end_effector" pos="...">
    <geom name="pusher_tip" type="cylinder" ... />
    <camera name="eye_in_hand" pos="0.03 0.03 0.05" quat="..." fovy="70" />
</body>
```

眼在手上相机的外参 (`data.cam_xpos` / `data.cam_xmat`) 会在每次 `mj_forward` / `mj_step` 后自动更新，无需手动处理。

---

#### 第 2 步：实现 `Task` 接口

创建你的 task 模块（可以放在项目外任意位置），继承 `Task` ABC：

```python
import numpy as np
from domain_rand.tasks.base import Task

class PushTTask(Task):
    """Push-T 任务：控制末端推动 T 形物体到目标位置。"""

    def __init__(self):
        # 配置参数
        self.obj_body = "push_t"
        self.goal_body = "goal"
        self.max_steps = 200

    # ── 必须实现 ──────────────────────────────────────

    def reset(self, scene, rng):
        """每 episode 开始时调用。随机化初始状态并返回初始观测。"""
        # 1. 随机化物体位姿
        obj_x = rng.uniform(-0.25, 0.25)
        obj_y = rng.uniform(-0.18, 0.18)
        obj_yaw = rng.uniform(-3.14, 3.14)
        qw, qz = np.cos(obj_yaw / 2), np.sin(obj_yaw / 2)
        scene.set_joint_qpos(self.obj_body,
            np.array([obj_x, obj_y, 0.43, qw, 0.0, 0.0, qz]))

        # 2. 随机化目标位置
        goal_x = rng.uniform(-0.20, 0.20)
        goal_y = rng.uniform(-0.15, 0.15)
        gid = scene.get_body_index(self.goal_body)
        scene.model.body_pos[gid] = np.array([goal_x, goal_y, 0.426])

        # 3. 重置机器人到零位
        scene.reset_dynamics()

        # 4. 让物理稳定（物体落到桌面）
        for _ in range(50):
            scene.step()

        self._step_count = 0
        return self.get_observation(scene)

    def step(self, scene, action):
        """执行一步。action 来自 Policy（遥操作/脚本/模型）。"""
        # 1. 施加动作到机器人关节
        qpos = scene.get_joint_qpos("j_x")  # 示例：读取 X 关节
        scene.set_joint_qpos("j_x", qpos + action[0:1])
        # ... 同理设置 Y 和 yaw 关节

        # 2. 步进仿真（可多次子步）
        for _ in range(5):
            scene.step()

        # 3. 计算 reward（物体离目标多远）
        obj_pos = scene.get_body_pose(self.obj_body)[0]
        goal_pos = scene.get_body_pose(self.goal_body)[0]
        dist = np.linalg.norm(obj_pos[:2] - goal_pos[:2])
        reward = -dist

        # 4. 判断终止
        self._step_count += 1
        done = (dist < 0.03) or (self._step_count >= self.max_steps)

        obs = self.get_observation(scene)
        info = {"distance": float(dist)}
        return obs, reward, done, info

    def get_observation(self, scene):
        """返回当前观测。DemoCollector 会自动补充渲染图像。"""
        return {
            "state": self._get_state_vector(scene),
        }

    # ── 可选 ──────────────────────────────────────────

    @property
    def action_spec(self):
        return {"shape": (3,), "dtype": "float32",
                "names": ["dx", "dy", "dyaw"]}

    @property
    def state_spec(self):
        return {"shape": (6,), "dtype": "float32",
                "names": ["jx", "jy", "jyaw", "obj_x", "obj_y", "obj_yaw"]}

    # ── 内部辅助 ──────────────────────────────────────

    def _get_state_vector(self, scene):
        jx = scene.get_joint_qpos("j_x")[0]
        jy = scene.get_joint_qpos("j_y")[0]
        jyaw = scene.get_joint_qpos("j_yaw")[0]
        obj_pos = scene.get_body_pose(self.obj_body)[0]
        return np.array([jx, jy, jyaw, obj_pos[0], obj_pos[1], 0.0],
                        dtype=np.float32)
```

每个方法的职责：

| 方法 | 调用时机 | 职责 |
|------|---------|------|
| `reset(scene, rng)` | 每 episode 开始 | 随机化初始状态、重置机器人、返回 `obs` |
| `step(scene, action)` | 每帧 | 施加 action → 步进仿真 → 计算 reward + done |
| `get_observation(scene)` | reset / step 后 | 返回 `{"state": np.ndarray}`，图像由框架补充 |
| `action_spec` / `state_spec` | 初始化 | 元数据，写入 HDF5 `.attrs` |

**重要**：`Scene` 提供了丰富的辅助方法，task 作者无需手动调用 `mujoco.mj_name2id` 或操作底层数组索引。见 [scene.py](src/domain_rand/core/scene.py) 中的 `get_joint_qpos` / `set_joint_qpos` / `get_body_pose` / `reset_dynamics` 等。

---

#### 第 3 步：配置 + 运行

创建配置文件（只需覆盖场景路径和相机名称）：

```yaml
# configs/my_push_task.yaml
scene_path: "assets/scenes/push_t_robot.xml"

il_demo:
  num_demos: 100
  max_steps: 300
  camera: "eye_in_hand"      # 眼在手上相机名称
  display_scale: 1.0

texture:
  enabled: true               # 纹理 DR 开启（训练鲁棒策略）
  exclude_geoms: [table_top, goal_]  # 排除不需要随机化的物体

lighting:
  enabled: true               # 光照 DR 开启

camera:
  enabled: false              # 眼在手上相机不应抖动
```

运行采集：

```bash
python scripts/collect_demos.py \
    --config configs/my_push_task.yaml \
    --task my_tasks.push_t.PushTTask \
    --policy domain_rand.policy.keyboard_teleop.KeyboardTeleop \
    -n 100 -o ./datasets/push_t_demos.h5
```

参数说明：

| 参数 | 说明 |
|------|------|
| `--config` | 场景 + DR + IL 配置文件 |
| `--task` | Task 类的完整模块路径（`module.ClassName`） |
| `--policy` | Policy 类的完整模块路径（内置 `KeyboardTeleop` 或自定义） |
| `-n` / `--num-demos` | 采集的 demo 数量 |
| `-o` / `--output` | 输出 HDF5 路径 |
| `-r` / `--recorder` | 覆盖 recorder 后端（`simple` / `robomimic` / 类路径） |
| `--recorder-kwargs` | JSON 字符串，覆盖 recorder 构造参数 |

### 键盘遥操作说明

内置 `KeyboardTeleop` 默认键位（可在代码中自定义）：

| 按键 | 动作维度 | 效果 |
|------|---------|------|
| `W` / `S` | dim 0 | X 轴正/负方向移动 |
| `A` / `D` | dim 1 | Y 轴正/负方向移动 |
| `Q` / `E` | dim 2 | 偏航角正/负旋转 |
| `N` | — | 提前结束当前 episode，进入下一个 |
| `R` | — | 重置当前 episode（丢弃数据重来） |
| `ESC` | — | 退出采集 |

可通过 `add_key_binding(key, dim, sign)` 或覆写 `DEFAULT_KEYMAP` 自定义键位。

### HDF5 输出格式定制（可插拔 Recorder）

通过配置 `il_demo.recorder` 字段，可以切换 HDF5 输出格式，无需修改 task 或 collector 代码。

#### 三种 Recorder

| recorder 值 | 类 | 说明 |
|-------------|-----|------|
| `"simple"` | `SimpleRecorder` | **默认**。灵活 dict 格式，Task 返回的所有 key 自动记录 |
| `"robomimic"` | `RobomimicRecorder` | robomimic 兼容格式，可直接用 `robomimic.SequenceDataset` 加载 |
| `"pkg.MyRecorder"` | 自定义 | 实现 `BaseRecorder` ABC，框架通过类路径动态加载 |

配置方式（YAML）：

```yaml
il_demo:
  recorder: "robomimic"                # 选择 recorder
  recorder_kwargs:                     # 传给 recorder 构造函数的额外参数
    image_keys: ["rgb"]                # 哪些 key 是图像（走 gzip 压缩）
    state_keys: ["state"]              # 哪些 key 拼接成 states 数据集
    key_rename:                        # 可选：输出 key 重命名
      rgb: "agentview_image"
```

CLI 覆盖：

```bash
python scripts/collect_demos.py \
    --recorder robomimic \
    --recorder-kwargs '{"image_keys":["rgb"],"state_keys":["state"]}' \
    ...
```

---

#### 格式 1：SimpleRecorder（默认）

Task 返回什么就记录什么，无需额外配置。

```
dataset.h5
├── .attrs/                     # 全局元数据
└── episode_XXXX/
    ├── observations/           # Task.get_observation() 的所有 key
    │   ├── rgb                # (T, H, W, 3)  ← 图像自动 gzip 压缩
    │   ├── state              # (T, D)
    │   ├── rgb_side           # (T, H, W, 3)  ← 自定义额外相机
    │   └── ...                # 任意其他 key
    ├── infos/                  # Task.step() 的 info dict（可选）
    │   ├── distance           # (T,)
    │   ├── success            # (T,)
    │   └── ...
    ├── actions                 # (T, A)
    ├── rewards                 # (T,)
    ├── dones                   # (T,)
    └── .attrs/meta             # episode 元数据 (JSON)
```

**数据自动流转规则**：
- `Task.get_observation()` 返回 `{"state": ..., "rgb_side": ...}` → 全部记入 `observations/`
- `Task.step()` 的 `info` 返回 `{"distance": 0.1}` → 全部记入 `infos/`
- 框架自动为 rank ≥ 3 的数组启用 gzip 压缩

---

#### 格式 2：RobomimicRecorder

输出符合 [robomimic](https://github.com/ARISE-Initiative/robomimic) 官方格式，可直接用于训练：

```
demo.hdf5
├── data/
│   ├── demo_0/
│   │   ├── states              # (T, D)  ← state_keys 拼接
│   │   ├── actions             # (T, A)
│   │   ├── rewards             # (T,)
│   │   ├── dones               # (T,)
│   │   ├── obs/                # 所有 observation key
│   │   │   ├── agentview_image # image_keys → gzip 压缩
│   │   │   ├── state           # 低维状态分量
│   │   │   └── ...
│   │   ├── model_file          # MuJoCo XML 字符串
│   │   └── .attrs/
│   └── ...
└── .attrs/
    ├── total
    └── env_args
```

`recorder_kwargs` 参数说明：

| 参数 | 类型 | 默认值 | 说明 |
|------|------|--------|------|
| `image_keys` | `list[str]` | `["rgb"]` | 图像类 observation key，存入 `obs/` 并 gzip 压缩 |
| `state_keys` | `list[str]` | `["state"]` | 低维状态 key，**按顺序拼接**为 `states` 数据集 |
| `key_rename` | `dict[str,str]` | `{}` | 输出时重命名 key（如 `rgb` → `agentview_image`） |
| `store_model_xml` | `bool` | `true` | 是否在每个 demo 中存储 MuJoCo XML |

**完整配置示例**（多相机 + 分体状态）：

```yaml
il_demo:
  recorder: "robomimic"
  recorder_kwargs:
    image_keys: ["rgb", "rgb_wrist"]
    state_keys: ["robot_joints", "object_pose"]
    key_rename:
      rgb: "agentview_image"
      rgb_wrist: "robot0_eye_in_hand_image"
      robot_joints: "robot0_joint_pos"
```

对应 Task 的 `get_observation()` 返回：
```python
def get_observation(self, scene):
    return {
        "rgb": ...,                 # → obs/agentview_image (gzip 压缩)
        "rgb_wrist": ...,           # → obs/robot0_eye_in_hand_image (gzip)
        "robot_joints": ...,        # → states 的第 1 段 + obs/robot0_joint_pos
        "object_pose": ...,         # → states 的第 2 段 + obs/object_pose
    }
```

---

#### 格式 3：自定义 Recorder

实现 `BaseRecorder` 三个方法，放在任意 Python 模块中，框架通过类路径动态加载。

**接口定义**（[recorders/base.py](src/domain_rand/pipeline/recorders/base.py)）：

```python
from domain_rand.pipeline.recorders.base import BaseRecorder

class MyRecorder(BaseRecorder):
    def __init__(self, output_path, **kwargs):
        # kwargs 来自 recorder_kwargs 配置
        ...

    def open(self, attrs: dict | None = None) -> None:
        """打开文件，写入全局属性。"""
        ...

    def write_episode(
        self,
        episode_idx: int,
        observations: dict[str, np.ndarray],  # {"rgb": (T,H,W,3), "state": (T,D), ...}
        actions: np.ndarray,                   # (T, A)
        rewards: np.ndarray,                   # (T,)
        dones: np.ndarray,                     # (T,)
        infos: dict[str, np.ndarray] | None,   # {"distance": (T,), ...}
        meta: dict | None,                     # episode 元数据
    ) -> None:
        """写入一个 episode 的完整轨迹。"""
        ...

    def close(self) -> None:
        """关闭文件。"""
        ...
```

**使用方式**：

```yaml
il_demo:
  recorder: "my_project.my_recorder.MyRecorder"
  recorder_kwargs:
    any_custom_param: 42
```

框架通过 `importlib` 加载 `my_project.my_recorder.MyRecorder`，传入 `output_path` 和 `recorder_kwargs`。

**最小实现示例**（写 JSONL 而非 HDF5）：

```python
import json
import numpy as np
from domain_rand.pipeline.recorders.base import BaseRecorder

class JSONLRecorder(BaseRecorder):
    def __init__(self, output_path, **kwargs):
        self.path = output_path
        self.indent = kwargs.get("indent", None)

    def open(self, attrs=None):
        self._f = open(self.path, "w")

    def write_episode(self, ep, obs, actions, rewards, dones, infos=None, meta=None):
        # 将 numpy 数组转为 list，写入一行 JSON
        record = {
            "episode": ep,
            "T": len(actions),
            "actions": actions.tolist(),
            "rewards": rewards.tolist(),
            "meta": meta,
        }
        self._f.write(json.dumps(record, indent=self.indent) + "\n")

    def close(self):
        self._f.close()
```

### Scene 辅助方法速查

Task 实现中常用的方法（[scene.py](src/domain_rand/core/scene.py)）：

| 方法 | 返回 | 用途 |
|------|------|------|
| `get_joint_qpos(name)` | `np.ndarray` | 读取关节位置 |
| `set_joint_qpos(name, val)` | — | 设置关节位置 |
| `get_joint_qvel(name)` | `np.ndarray` | 读取关节速度 |
| `get_body_pose(name)` | `(pos, quat)` | 读取 body 世界位姿 |
| `get_body_index(name)` | `int` | body 索引（用于读写 `model.body_pos[id]`） |
| `get_joint_index(name)` | `int` | 关节索引 |
| `get_camera_pose(name)` | `(pos, rot_mat)` | 相机世界位姿 |
| `reset_dynamics()` | — | qpos/qvel/ctrl 清零 + forward |
| `forward()` | — | 正向运动学（不积分） |
| `step()` | — | 完整仿真步（积分 + 碰撞） |
| `model` | `MjModel` | 原始 MuJoCo 模型（高级用法） |
| `data` | `MjData` | 原始 MuJoCo 数据（高级用法） |

## 扩展：新增随机化器

继承 `Randomizer` 基类：

```python
from domain_rand.randomizers.base import Randomizer

class MyRandomizer(Randomizer):
    def save_nominal(self, model): ...
    def randomize(self, model): ...
    def restore(self, model): ...
```

然后在 `DomainRandomizer` 中注册即可。

