# RGB-D 深度补全与抓取检测

本项目把 RGB-D 图像依次交给所选深度模型和 AnyGrasp，生成可供机械臂使用的抓取位姿：

```text
离线 RGB-D 图像或 Intel RealSense
  → 所选深度模型完成深度预测
  → AnyGrasp 2026 抓取检测
  → 抓取位姿文件
  → 可选 Piper 机械臂执行
```

项目支持两个 checkpoint 架构：`defm_vit_l14_depth`（DeFM ViT-L/14 + DPT decoder）和
`defm_stackconv_depth`（DeFM ViT-L/14 + stack-conv decoder）。模型架构由 `--depth-model` 明确选择，
程序不会根据 checkpoint 内容猜测架构。默认只运行感知并保存结果，**不会控制机械臂**；只有显式添加
`--execute-arm` 才会执行 Piper 控制代码。

## 先选择你的使用方式

| 目标 | 推荐入口 | 平台 | 需要准备 |
| --- | --- | --- | --- |
| 只验证深度模型 | `asdepth_depth_only.py` | macOS 或 Linux | 深度模型 checkpoint、RGB-D 图像 |
| 用离线图像检测抓取 | `asdepth_pipeline.py` | Linux x86-64 | 两个 checkpoint、AnyGrasp 许可证、完整运行环境 |
| 用 RealSense 在线检测 | `asdepth_pipeline.py` | Linux x86-64 | 上述内容和 Intel RealSense |
| 控制 Piper 抓取 | `asdepth_pipeline.py --execute-arm` | 现场 Linux 主机 | 上述内容、Piper、CAN、手眼标定和安全确认 |

推荐第一次使用时严格按照下面的顺序操作：

1. 先用示例图片完成“仅深度预测”；
2. 再在 Linux 上完成“离线抓取检测”；
3. 然后连接 RealSense 完成在线检测；
4. 最后在完成标定、限位和现场安全检查后启用机械臂。

## 1. 环境要求

### 仅深度预测

- macOS 或 Linux；
- 推荐 Python 3.10；
- CPU、Apple MPS 或 NVIDIA CUDA 均可尝试；
- 不需要 AnyGrasp、RealSense、许可证或 Piper。

### 完整抓取链路

- Linux x86-64；
- 推荐 Python 3.10；
- NVIDIA CUDA 和与之匹配的 PyTorch；
- MinkowskiEngine 以及 AnyGrasp 所需的系统依赖；
- 新版 AnyGrasp 许可证；
- 连接相机时需要 `pyrealsense2`；
- 控制机械臂时还需要 Piper SDK、CAN 接口和正确的手眼标定。

仓库中的 AnyGrasp GSNet 二进制覆盖多个 CPython ABI，但 CUDA、PyTorch、MinkowskiEngine、glibc 和其他依赖仍需彼此兼容。普通用户建议统一使用 Python 3.10，减少环境组合问题。

> macOS 可以运行深度预测，但不能加载仓库中的 Linux x86-64 AnyGrasp 二进制。

## 2. 使用 Conda 创建环境

本文默认推荐 [Miniconda](https://docs.conda.io/projects/miniconda/en/latest/) 或 Anaconda。进入仓库根目录后创建独立环境：

```bash
cd /path/to/grasp_detection_asdepth

conda create -n grasp-asdepth python=3.10 -y
conda activate grasp-asdepth

python -m pip install --upgrade pip setuptools wheel
```

以后每次重新打开终端，都需要先进入项目并激活环境：

```bash
cd /path/to/grasp_detection_asdepth
conda activate grasp-asdepth
```

### 方案 A：只运行深度预测

```bash
python -m pip install -r requirements-asdepth.txt
```

### 方案 B：运行完整抓取链路

如果需要指定 CUDA 版本，请先按照 [PyTorch 官方安装说明](https://pytorch.org/get-started/locally/) 安装匹配的 PyTorch，再安装项目依赖：

```bash
python -m pip install -r requirements.txt
python -m pip install -r requirements-asdepth.txt
python -m pip install -r requirements-realsense.txt
```

MinkowskiEngine 与 CUDA、PyTorch 的版本耦合较强，因此没有写入通用 requirements。请按照 AnyGrasp SDK 和 MinkowskiEngine 的对应版本说明安装，然后检查环境：

```bash
python -c "import torch; print('torch:', torch.__version__); print('cuda available:', torch.cuda.is_available())"
python -c "import MinkowskiEngine; print('MinkowskiEngine: OK')"
```

即使使用离线 RGB-D 图片，`asdepth_pipeline.py` 当前也会通过 `get_pose.py` 导入 `pyrealsense2`，所以完整抓取环境仍需安装 `requirements-realsense.txt`。

`install.sh` 仍可作为基于 `venv` 的备选安装方式，但它只安装基础依赖；Conda 是本文推荐和默认说明的环境管理方式。

## 3. 准备模型文件

模型权重不包含在 Git 仓库中。请从项目维护者或对应上游取得所选深度模型和 AnyGrasp 的权重文件：

| 文件 | 用途 | 示例路径 |
| --- | --- | --- |
| 深度模型 checkpoint | 与 `--depth-model` 所选架构匹配的米制深度模型 | `ckpts/depth_model.ckpt` |
| AnyGrasp checkpoint | 从点云预测抓取位姿 | `ckpts/checkpoint-rs.tar` |

建议统一放到仓库的 `ckpts/` 目录：

```bash
mkdir -p ckpts
cp /path/to/depth_model.ckpt ckpts/depth_model.ckpt
cp /path/to/checkpoint-rs.tar ckpts/checkpoint-rs.tar
```

`ckpts/` 已被 `.gitignore` 忽略，不会被误提交。程序支持 `defm_vit_l14_depth` 和
`defm_stackconv_depth`，并按 `--depth-model` 构建对应模型；权重与所选架构不一致时严格加载会失败。

默认使用安全的权重读取方式。如果必须加载旧式 pickle checkpoint，只能在确认文件来源可信时添加 `--trusted-depth-checkpoint`。

## 4. 准备 AnyGrasp 许可证

本节只需要在运行完整抓取链路时执行，并且必须在 Linux x86-64 目标机上完成。

先确认程序找到了与当前 Python 匹配的 GSNet 二进制：

```bash
python -c "from anygrasp_runtime import matching_gsnet_path; print(matching_gsnet_path())"
```

获取本机 feature ID：

```bash
python -c "from anygrasp_runtime import load_gsnet_module; print(load_gsnet_module().get_feature_id())"
```

按照 AnyGrasp 上游流程申请新版许可证，然后把许可证目录放到仓库根目录：

```text
grasp_detection_asdepth/
├── license/
│   └── licenseCfg.json
├── anygrasp_runtime.py
└── ...
```

验证许可证：

```bash
python -c "from anygrasp_runtime import load_gsnet_module; load_gsnet_module().check_license('license'); print('license check finished')"
```

`license/` 已被 `.gitignore` 忽略。旧的 `lib_cxx.so` 和旧许可证不能用于当前 AnyGrasp 2026 接口。

## 5. 准备 RGB-D 输入

离线入口需要一张彩色图和一张原始深度图：

- 彩色图：常规三通道图片，例如 PNG；
- 深度图：单通道图片，推荐保存为 16 位 PNG；
- 两张图片的宽高必须完全一致；
- 默认认为深度值单位是毫米，因此使用 `--depth-scale 1000` 转成米；
- 深度值 `0`、非有限值和大于等于 `--max-depth` 的值会被当作无效点。

仓库提供了可用于检查流程的输入：

```text
example_data/color.png
example_data/depth.png
```

如果相机的每个深度单位代表 `s` 米，则应设置 `--depth-scale 1/s`。例如深度单位为 `0.001 m` 时，使用默认值 `1000`。

## 6. 先运行仅深度预测

这是推荐的第一次运行方式，不会导入 AnyGrasp、RealSense 或 Piper：

```bash
python asdepth_depth_only.py \
  --depth-checkpoint ckpts/depth_model.ckpt \
  --depth-model defm_vit_l14_depth \
  --rgb-image example_data/color.png \
  --depth-image example_data/depth.png \
  --save-dir debug/asdepth-only \
  --device auto
```

`--device auto` 会依次尝试 CUDA、Apple MPS 和 CPU。也可以显式指定：

```bash
# Apple Silicon
python asdepth_depth_only.py ... --device mps

# 纯 CPU
python asdepth_depth_only.py ... --device cpu

# NVIDIA GPU
python asdepth_depth_only.py ... --device cuda
```

程序成功后会在终端打印本次运行目录，并生成：

```text
debug/asdepth-only/run_日期_时间/
├── pred_depth.npy
└── run_metadata.json
```

- `pred_depth.npy`：与输入图像同尺寸的 `float32` 深度数组，单位为米；
- `run_metadata.json`：模型、输入路径、设备、参数和耗时。

如果 MPS 上遇到算子兼容问题，可先改用 `--device cpu` 判断是否为设备问题。

## 7. 配置相机内参和抓取工作区

在运行 AnyGrasp 前，必须把 [get_pose.py](get_pose.py) 中 `run_anygrasp()` 使用的参数改成现场相机和工作区参数：

```python
# 相机内参，必须对应实际分辨率
fx, fy = ...
cx, cy = ...

# 相机坐标系下的抓取范围，单位为米
xmin, xmax = ...
ymin, ymax = ...
zmin, zmax = ...
```

当前 RealSense 采集分辨率固定为 `640 × 480`，彩色流和深度流均为 30 FPS，深度会对齐到彩色图。内参必须对应对齐后的彩色相机分辨率。

工作区用于排除桌面外、机械臂不可达或不希望抓取的点。如果内参、深度比例或工作区不正确，常见结果是点云位置错误、没有有效点或抓取位姿明显偏移。

## 8. 使用离线 RGB-D 运行完整抓取检测

完成 Linux、CUDA、MinkowskiEngine、许可证和两个 checkpoint 的准备后运行：

```bash
python asdepth_pipeline.py \
  --depth-checkpoint ckpts/depth_model.ckpt \
  --depth-model defm_vit_l14_depth \
  --grasp-checkpoint ckpts/checkpoint-rs.tar \
  --rgb-image example_data/color.png \
  --depth-image example_data/depth.png \
  --save-dir debug/asdepth \
  --device cuda
```

这个命令只生成深度和抓取位姿，不会控制机械臂。需要查看 Open3D 点云和抓取候选时，可以添加 `--debug`：

```bash
python asdepth_pipeline.py \
  --depth-checkpoint ckpts/depth_model.ckpt \
  --depth-model defm_vit_l14_depth \
  --grasp-checkpoint ckpts/checkpoint-rs.tar \
  --rgb-image example_data/color.png \
  --depth-image example_data/depth.png \
  --save-dir debug/asdepth \
  --device cuda \
  --debug
```

服务器没有图形桌面时不要添加 `--debug`，否则 Open3D 窗口可能无法创建。

## 9. 使用 RealSense 在线检测

连接 RealSense 后，不传 `--rgb-image` 和 `--depth-image` 即可采集一帧并继续推理：

```bash
python asdepth_pipeline.py \
  --depth-checkpoint ckpts/depth_model.ckpt \
  --depth-model defm_vit_l14_depth \
  --grasp-checkpoint ckpts/checkpoint-rs.tar \
  --save-dir debug/asdepth \
  --device cuda
```

程序会等待约 30 帧让相机稳定，然后保存对齐后的 `color.png` 和 `depth.png`。如果设备报告的深度单位不是 `0.001 m`，请根据实际比例传入正确的 `--depth-scale`。

如果程序找不到相机，请先确认：

- USB 连接和供电正常；
- 系统已经安装 Intel RealSense SDK；
- 当前用户有访问相机设备的权限；
- `python -c "import pyrealsense2"` 可以成功执行。

## 10. 查看运行结果

离线运行会创建 `run_...` 目录，RealSense 运行会创建 `capture_...` 目录。完整流程的主要输出如下：

```text
debug/asdepth/<本次运行目录>/
├── color.png            # RealSense 模式生成
├── depth.png            # RealSense 模式生成的原始深度
├── pred_depth.npy       # 深度模型输出，float32，单位米
├── grasp_pose.txt       # 最佳抓取的旋转、平移和夹爪宽度
├── run_metadata.json    # 参数、文件路径、设备和耗时
└── scene_cloud_14b.ply  # 仅 --debug 时可能生成
```

`grasp_pose.txt` 中：

- `R_cam` 是相机坐标系下的 `3 × 3` 旋转矩阵；
- `t_cam` 是相机坐标系下的三维位置，单位为米；
- `width` 是建议夹爪开口宽度，单位为米。

建议先检查 `pred_depth.npy`、点云方向和 `grasp_pose.txt`，确认结果合理后再考虑机械臂执行。

## 11. 控制 Piper 机械臂

> **危险操作：`--execute-arm` 会真实连接 `can0` 并发送机械臂运动和夹爪指令。未经标定和现场确认不要使用。**

启用前至少完成以下检查：

1. 在 [grasp_piper.py](grasp_piper.py) 中替换现场标定得到的 `T_cam2base`；
2. 检查 `GRIPPER_DISTANCE`、预抓取距离、固定初始位姿和姿态轴映射；
3. 确认机械臂 CAN 接口确实是 `can0`，并已正确启动；
4. 检查工作区、机械臂可达范围、速度、碰撞风险和急停；
5. 先在不带 `--execute-arm` 的模式下反复验证抓取位姿；
6. 首次执行时清空工作区，并由操作人员随时准备急停。

确认后，在线抓取命令为：

```bash
python asdepth_pipeline.py \
  --depth-checkpoint ckpts/depth_model.ckpt \
  --depth-model defm_vit_l14_depth \
  --grasp-checkpoint ckpts/checkpoint-rs.tar \
  --save-dir debug/asdepth \
  --device cuda \
  --execute-arm
```

机械臂是否实际执行会记录在 `run_metadata.json` 的 `arm_executed` 字段中。

## 12. 常用参数

| 参数 | 默认值 | 说明 |
| --- | --- | --- |
| `--depth-model` | 必填 | 显式选择 `defm_vit_l14_depth` 或 `defm_stackconv_depth` |
| `--device` | `auto` | 深度模型推理设备，例如 `cuda`、`mps` 或 `cpu` |
| `--depth-scale` | `1000` | 原始深度值除以该数后得到米 |
| `--max-depth` | `10` | 深度有效上限，单位米 |
| `--input-size` | `518` | 深度模型预处理目标尺寸 |
| `--resize-method` | `lower_bound` | 预处理缩放方式 |
| `--max-gripper-width` | `0.1` | AnyGrasp 接受的最大夹爪宽度，单位米 |
| `--gripper-height` | `0.03` | 夹爪高度，单位米 |
| `--top-down-grasp` | 开启 | 优先输出俯视抓取；可用 `--no-top-down-grasp` 关闭 |
| `--debug` | 关闭 | 打开 Open3D 可视化并保存调试点云 |
| `--execute-arm` | 关闭 | 真实执行 Piper 控制，属于危险操作 |
| `--trusted-depth-checkpoint` | 关闭 | 允许读取可信来源的旧式 pickle checkpoint |

查看脚本支持的全部参数：

```bash
python asdepth_depth_only.py --help
python asdepth_pipeline.py --help
```

## 13. 常见问题

### `AnyGrasp GSNet requires Linux x86-64`

当前系统不是 Linux x86-64。macOS 只能运行 `asdepth_depth_only.py`，不能运行 AnyGrasp。

### `no matching AnyGrasp GSNet binary`

当前 Python ABI 没有匹配的二进制。优先改用推荐的 Conda Python 3.10 环境，并确认 `gsnet_versions/` 文件完整。

### `AnyGrasp license is missing`

许可证目录位置不正确。必须是仓库根目录下的 `license/licenseCfg.json`。

### 导入 GSNet 时提示缺少 `.so`、CUDA 或 MinkowskiEngine

这是二进制运行环境不匹配。检查 Linux/glibc、CUDA、PyTorch、MinkowskiEngine 和 GSNet 是否使用兼容版本。

### 离线完整流程仍提示缺少 `pyrealsense2`

这是当前代码的已知依赖边界：完整入口会导入 `get_pose.py`。安装 `requirements-realsense.txt`；如果所在平台无法安装，请只运行深度入口，或改到支持 RealSense Python 包的 Linux 环境。

### 深度模型 checkpoint 不兼容

请确认 `--depth-model` 与权重对应：DeFM + DPT 权重使用 `defm_vit_l14_depth`，
DeFM + stack-conv 权重使用 `defm_stackconv_depth`，同时确认没有误传 AnyGrasp checkpoint。

### `No valid point remains` 或工作区内没有点

依次检查 RGB/深度尺寸、`--depth-scale`、相机内参、深度有效范围和 `get_pose.py` 中的工作区边界。

### 没有检测到抓取

先确认点云方向和尺度正确，再适当调整工作区、`--max-gripper-width` 或 `--no-top-down-grasp`。不要通过扩大机械臂执行范围来掩盖点云或标定问题。

### 出现 `xFormers is disabled` 或 `xFormers is not available`

DeFM 桥接会主动使用非 xFormers 路径以保持 checkpoint 参数结构兼容。这类警告本身不代表推理失败；以最终是否成功生成输出为准。

## 14. 其他入口

- `asdepth_depth_only.py`：推荐的深度模型离线验证入口；
- `asdepth_pipeline.py`：推荐的完整抓取入口；
- `demo.py` / `demo.sh`：只使用 AnyGrasp 的离线场景示例，读取 `test_color.png` 和 `test_depth.png`，不经过深度补全模型；
- `get_pose.py`：RealSense 采集和 AnyGrasp 的底层实现，包含现场相机内参与工作区；
- `grasp_piper.py`：Piper 坐标转换和控制逻辑；
- `USAGE.md`：AnyGrasp 2026 SDK 接口和 steering 参数说明。

## 15. 测试

在 Conda 环境中运行：

```bash
conda activate grasp-asdepth
python -m pip install -r requirements-dev.txt
python -m unittest discover -s tests -v
ruff check .
ruff format --check .
```

Ruff 规则集中配置在 `pyproject.toml` 中，并排除了按上游原样保留的 vendor 代码。单元测试不会加载真实 checkpoint、GSNet 或 RealSense，也不会控制机械臂。真实 CUDA、许可证、相机和机械臂仍需在目标 Linux 主机上分别验收。

## 第三方代码与边界

AnyGrasp SDK 资产基于上游 `graspnet/anygrasp_sdk` 的 2026 接口，本项目通过 `anygrasp_runtime.py` 处理二进制选择、许可证和 steering 适配。

`asdepth_depth/` 只迁入 `defm_vit_l14_depth` 和 `defm_stackconv_depth` 推理所需的代码，并包含来自 ByteDance Seed CDM、Meta DINOv2、ETH Zurich DeFM 和 MoGe 的派生实现。训练、评估、数据集、其他模型架构和模型权重均不包含在本仓库中。版权与许可证说明见 `ASDEPTH_NOTICE` 和 `ASDEPTH_LICENSE`。
