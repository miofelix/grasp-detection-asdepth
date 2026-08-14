# grasp_detection

本项目是一套可维护的 RGB-D 抓取流水线，负责连接深度相机、可选择的深度补全模型、
AnyGrasp 抓取检测以及可选的 Piper 机械臂控制。仓库同时保留历史业务入口，方便已有部署继续运行，
并提供基于统一模型目录的新推理入口，便于后续增加模型、升级 SDK 或替换硬件。

## 系统组成

推荐使用的完整数据流为：

```text
RealSense 或离线 RGB-D
  → AS-Depth 模型目录中的深度补全模型
  → float32 米制深度图
  → AnyGrasp 抓取检测
  → 抓取位姿文件
  → 可选 Piper 机械臂执行
```

各部分的维护边界如下：

| 模块 | 职责 | 主要入口 |
|---|---|---|
| RGB-D 输入 | RealSense 对齐采集或离线图像读取 | `get_pose.py`、`asdepth_pipeline.py` |
| 深度模型 | 模型选择、checkpoint 加载、RGB-D 推理和深度语义转换 | `asdepth_depth/`、`asdepth_depth_only.py` |
| 抓取检测 | 调用 AnyGrasp 公开接口并传入项目配置 | `anygrasp_runtime.py`、`get_pose.run_anygrasp` |
| 场景适配 | 相机内参、workspace、数据命名和启动参数 | `demo.py`、`demo.sh` |
| 机械臂 | 抓取位姿转换与 Piper 执行 | `grasp_piper.py` |
| 历史流程 | 远程单目深度、AprilTag、AnyGrasp 和 Piper | `pipline.py` |

## 上游来源与代码边界

### AnyGrasp SDK

AnyGrasp 的官方上游仓库是
[graspnet/anygrasp_sdk](https://github.com/graspnet/anygrasp_sdk)。当前 SDK 资产基于上游
`b8eaafc9eca7babd5208e7a5ade3c561060be4c5` 快照，`grasp_detection` 使用的二进制兼容更新对应
`ada42fa`。

本仓库不直接修改官方 SDK 的实现逻辑。项目差异通过以下边界维护：

- `anygrasp_runtime.py`：加载与当前 Python ABI 匹配的 GSNet 二进制、检查许可证，并调用公开接口；
- `get_pose.py`：将 RGB-D 数据和项目参数传给 AnyGrasp；
- `demo.py`、`demo.sh`：项目自有的场景入口，保留 D435 内参、数据前缀和 workspace 等配置；
- `USAGE.md`：保留随 SDK 同步的官方使用说明。

升级 AnyGrasp 时，应先同步官方资产，再在上述项目接口层处理配置差异，避免把场景逻辑写入官方代码。

### 深度模型包

深度模型与推理接口来自
[miofelix/AS-Depth-Research](https://github.com/miofelix/AS-Depth-Research)。仓库内置的
`vendor/as_depth-0.3.0-py3-none-any.whl` 构建自 commit
`214e501321634d44d2bccdaea4c5a7a637ee70bc`，用于提供：

- canonical 模型目录和稳定模型身份；
- 统一 checkpoint loader；
- metric depth 与 inverse depth 语义；
- RGB-D 内存推理；
- sparse-raw 和多任务模型的公共推理行为。

checkpoint、训练数据和运行产物不进入版本库。`asdepth_depth/models/` 是早期
`defm_stackconv_depth` 集成留下的兼容快照；新增模型或新推理功能应通过 `asdepth` 公共 API 接入，
不应继续扩展该快照。

## 支持的深度模型

当前模型目录包含 20 个活跃 `model_id`，覆盖以下主要架构族：

| 架构族 | 代表 model_id | 原生深度输出 |
|---|---|---|
| 双 DINOv2 CDM | `cdm_vitl_depth`、`cdm_vitl_disp`、D435/L515 pretrained | metric / inverse depth |
| DeFM ViT-L14 | `defm_vit_l14_depth`、`defm_vit_l14_random_raw` | metric depth |
| DINOv2 stack-conv | `stackconv_depth_multidataset`、`stackconv_depth_multidataset_new_fusion` | metric depth |
| DeFM stack-conv | `defm_stackconv_depth`、`defm_stackconv_no_adaptor_depth`、sparse-raw | metric depth |
| DeFM auxiliary/multitask | `defm_stackconv_mask_normal_depth`、`defm_stackconv_multitask_depth` | depth + auxiliary heads |

列出当前安装包实际提供的模型：

```bash
python asdepth_models.py
python asdepth_models.py --json
```

深度-only 与完整抓取入口均使用：

```text
--depth-checkpoint <checkpoint 文件、envelope 目录或 Accelerate 目录>
--depth-model <canonical model_id | auto>
```

默认模型为 `defm_stackconv_depth`，用于保持已有部署的 checkpoint 兼容性。指定其他
`model_id` 即可切换架构；`--depth-model auto` 会优先读取 checkpoint manifest，其次查询历史路径
registry。普通单文件 checkpoint 如果无法由 registry 识别，应显式指定 `model_id`。

适配层会根据模型声明转换 raw depth 和预测结果。无论模型原生使用 metric depth 还是 inverse depth，
传给 AnyGrasp 的结果始终是与相机帧同尺寸、finite、非负、`float32`、单位为 meter 的二维数组。
完整模型列表与 checkpoint 契约见 [docs/ASDEPTH_MODELS.md](docs/ASDEPTH_MODELS.md)。

## 环境要求

仅运行深度模型时可使用 Linux、macOS 或其他能够安装 PyTorch 的环境。完整抓取链路需要：

- Linux x86-64；
- CPython 3.10（推荐，便于与当前依赖和部署环境保持一致）；
- CUDA、MinkowskiEngine 和 AnyGrasp 对应的系统依赖；
- RealSense 场景需要 `pyrealsense2` 和相机驱动；
- 有效的新版 AnyGrasp 许可证；
- 外部提供的深度模型与 AnyGrasp checkpoint。

仓库不绑定 Python 环境管理器。可以使用标准 `venv`、Conda 或其他兼容环境，例如：

```bash
python3.10 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
python -m pip install -r requirements-asdepth.txt
python -m pip install -r requirements-realsense.txt
```

也可以使用仓库脚本创建标准 venv 并安装基础依赖：

```bash
bash install.sh --venv
source .venv/bin/activate
python -m pip install -r requirements-asdepth.txt
python -m pip install -r requirements-realsense.txt
```

不传 `--venv` 时，`install.sh` 会使用当前已激活的 Python 环境。该脚本只安装基础依赖，深度模型与
RealSense 依赖仍按运行场景单独安装。

只验证深度预测时，只需安装 `requirements-asdepth.txt`。完整抓取入口会复用 `get_pose.py`，而该模块
加载时会导入 `pyrealsense2`，因此当前离线抓取模式也需要安装 RealSense Python 依赖。

建议将 checkpoint 放入未跟踪的 `ckpts/`，也可以通过命令行传入任意外部路径。模型权重、许可证、
相机数据与运行结果不应提交到仓库。

## AnyGrasp 许可证

新版 GSNet 不再使用仓库历史版本的 `lib_cxx.so` 和旧许可证。先在目标 Linux 机器获取 feature ID：

```bash
python -c "from anygrasp_runtime import load_gsnet_module; print(load_gsnet_module().get_feature_id())"
```

按照 [AnyGrasp 官方仓库](https://github.com/graspnet/anygrasp_sdk) 的流程申请许可证，然后将许可证
目录放在仓库根目录的 `license/`。目录中必须包含 `licenseCfg.json`；`license/` 已被
`.gitignore` 排除。验证命令：

```bash
python -c "from anygrasp_runtime import load_gsnet_module; load_gsnet_module().check_license('license')"
```

## 运行方式

### 查询模型目录

```bash
python asdepth_models.py
```

### 仅运行深度预测

该入口不导入 AnyGrasp、RealSense 或 Piper，适合在开发机上检查 checkpoint、预处理和深度输出：

```bash
python asdepth_depth_only.py \
  --depth-checkpoint /path/to/model.ckpt \
  --depth-model defm_stackconv_depth \
  --rgb-image example_data/color.png \
  --depth-image example_data/depth.png \
  --save-dir debug/asdepth-only \
  --device auto
```

输出包括 `pred_depth.npy` 和 `run_metadata.json`。macOS（包括 Apple Silicon）可以运行该入口，
但不能加载仓库中的 Linux x86-64 AnyGrasp 二进制。

### 离线 RGB-D 抓取检测

```bash
python asdepth_pipeline.py \
  --depth-checkpoint /path/to/model.ckpt \
  --depth-model defm_stackconv_depth \
  --grasp-checkpoint ckpts/checkpoint-rs.tar \
  --rgb-image example_data/color.png \
  --depth-image example_data/depth.png \
  --save-dir debug/asdepth
```

### RealSense 在线采集

不传 `--rgb-image` 和 `--depth-image` 时，入口通过 `capture_one_frame()` 采集一组对齐 RGB-D：

```bash
python asdepth_pipeline.py \
  --depth-checkpoint /path/to/model.ckpt \
  --depth-model defm_stackconv_depth \
  --grasp-checkpoint ckpts/checkpoint-rs.tar \
  --save-dir debug/asdepth
```

默认只运行感知链路并保存抓取位姿，不控制机械臂。确认相机外参、工作空间和现场安全后，才应显式添加：

```bash
python asdepth_pipeline.py ... --execute-arm
```

### 项目场景 demo

`demo.py` 和 `demo.sh` 是本项目维护的场景适配入口，不等同于 AnyGrasp 官方 demo。默认配置可以通过
`CHECKPOINT_PATH`、`DATA_DIR` 或附加命令行参数覆盖：

```bash
CHECKPOINT_PATH=ckpts/checkpoint-rs.tar DATA_DIR=example_data bash demo.sh
```

## 运行产物

新流水线为每次执行创建独立目录：

```text
<run-dir>/
├── pred_depth.npy       # float32、meter、原相机尺寸
├── grasp_pose.txt       # 完整抓取流程生成
└── run_metadata.json    # 模型身份、checkpoint、参数和耗时
```

raw depth 默认按毫米解析（`--depth-scale 1000`），`0` 和 `>=10 m` 视为无效值。默认预处理使用
`input_size=518`、`lower_bound` 和 14 像素 patch 对齐，预测恢复到相机原始尺寸后再交给 AnyGrasp。

## 历史入口

- `pipline.py`：远程单目深度、AprilTag、AnyGrasp 和 Piper 的历史流程；
- `get_pose.py`：RealSense RGB-D 采集和 AnyGrasp 调用；
- `grasp_piper.py`：抓取位姿转换与 Piper 控制。

历史入口仍用于兼容已有部署。新功能应优先放入明确的适配层或新入口，避免扩大历史脚本与第三方代码的
耦合范围。

## 维护与升级

更新上游依赖时建议遵循以下顺序：

1. 在独立分支记录当前可运行版本，保留可回退点；
2. 记录上游仓库 URL、完整 commit 和二进制校验值；
3. 保持 AnyGrasp 官方实现不变，将相机、workspace 和业务参数放在项目接口层；
4. AS-Depth 新模型通过公共 catalog 和 inference API 接入，不直接扩展旧兼容快照；
5. 更新 README、模型目录说明、第三方 notice 和相关测试；
6. 在开发机运行轻量回归检查，并在目标 Linux/CUDA/RealSense 环境完成硬件验收。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试覆盖模型目录、模型入口导入、metric/inverse depth 转换、离线深度输出、AnyGrasp 接口适配和
机械臂安全开关。测试不会加载真实 checkpoint、GSNet 或控制机械臂。

以下项目必须在目标环境单独验证：

- 真实深度 checkpoint 的严格加载、显存占用、速度和数值结果；
- CUDA、MinkowskiEngine、glibc 与 GSNet 二进制兼容性；
- RealSense 对齐、相机内参与工作空间；
- AnyGrasp 许可证；
- Piper 坐标转换和机械臂安全边界。

## 第三方代码

`vendor/as_depth-0.3.0-py3-none-any.whl` 与 `asdepth_depth/` 中的兼容代码来自 AS-Depth，并包含
ByteDance Seed CDM、Meta DINOv2、ETH Zurich DeFM 和 MoGe 的派生实现。来源、范围与校验值见
[vendor/README.md](vendor/README.md)、[ASDEPTH_NOTICE](ASDEPTH_NOTICE) 和
[ASDEPTH_LICENSE](ASDEPTH_LICENSE)。这些内容与闭源 AnyGrasp SDK 及其许可证相互独立。
