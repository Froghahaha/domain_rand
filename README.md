# Domain Rand — MuJoCo 域随机化数据集生成工具

基于 MuJoCo 的 Domain Randomization 数据集制作工具，支持视觉域随机化（纹理/颜色、光照、相机），渲染 RGB + 深度图像，并以结构化 HDF5 格式导出数据集。

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
│   │   └── scene.py            # 场景管理
│   ├── randomizers/            # 随机化器
│   │   ├── base.py             # 抽象基类 + 组合器
│   │   ├── texture.py          # 纹理/颜色随机化
│   │   ├── lighting.py         # 光照随机化
│   │   └── camera.py           # 相机随机化
│   ├── pipeline/               # 数据采集管线
│   │   ├── collector.py        # 主采集循环
│   │   ├── recorder.py         # HDF5 记录器
│   │   └── metadata.py         # 元数据管理
│   └── utils/
│       └── rendering.py        # RGB/深度渲染 + 相机矩阵
├── assets/scenes/              # MuJoCo 场景文件
│   └── example_factory.xml     # 示例场景（箱子+物体）
├── scripts/                    # 运行脚本
│   ├── generate_dataset.py     # 数据集生成入口
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

## 扩展

新增随机化器只需继承 `Randomizer` 基类：

```python
from domain_rand.randomizers.base import Randomizer

class MyRandomizer(Randomizer):
    def save_nominal(self, model): ...
    def randomize(self, model): ...
    def restore(self, model): ...
```

然后在 `DomainRandomizer` 中注册即可。

## 许可

MIT
