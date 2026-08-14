# grasp_detection

本仓库保留师兄原有的 AnyGrasp、点云和 Piper 控制代码，同时提供可选择模型的本地 AS-Depth
RGB-D 推理入口。旧的 `pipline.py` 未修改，仍对应原来的远程 GAVP + AprilTag 流程。

AnyGrasp SDK 资产同步自 `/Users/felix/Projects/anygrasp_sdk` 的 `b8eaafc` 快照；其中
`grasp_detection` 的最后一项二进制更新为 `ada42fa`（glibc compatibility）。官方 SDK
源码不在本仓库修改，本项目通过 `anygrasp_runtime.py` 调用其公开接口并注入自己的相机与工作区配置。

## AS-Depth 多模型抓取入口

新入口的数据流为：

```text
RealSense 对齐 RGB-D
  → AS-Depth canonical catalog 模型
  → meter 深度图
  → 现有 get_pose.run_anygrasp
  → 可选 grasp_piper.run_pipline
```

仓库内置由 AS-Depth Research `214e501` 构建的 `as-depth 0.3.0` wheel，使用其 canonical
模型 catalog、checkpoint loader 和内存 RGB-D 推理接口。checkpoint、训练数据和运行产物仍由外部提供。
原有 `asdepth_depth/models` 仅作为早期 AS-Depth-2 迁移的兼容快照；新模型必须通过正式 catalog 选择。

### 支持的模型架构

当前 catalog 包含 20 个活跃 `model_id`，主要覆盖：

| 架构族 | 代表 model_id | 原生输出 |
|---|---|---|
| 双 DINOv2 CDM | `cdm_vitl_depth`、`cdm_vitl_disp`、D435/L515 pretrained | metric / inverse depth |
| DeFM ViT-L14 | `defm_vit_l14_depth`、`defm_vit_l14_random_raw` | metric depth |
| DINOv2 stack-conv | `stackconv_depth_multidataset`、`stackconv_depth_multidataset_new_fusion` | metric depth |
| DeFM stack-conv | `defm_stackconv_depth`、`defm_stackconv_no_adaptor_depth`、sparse-raw | metric depth |
| DeFM auxiliary/multitask | `defm_stackconv_mask_normal_depth`、`defm_stackconv_multitask_depth` | depth + auxiliary heads |

查看完整 catalog：

```bash
python asdepth_models.py
python asdepth_models.py --json
```

模型身份、完整列表、深度语义和 checkpoint 格式见 `docs/ASDEPTH_MODELS.md`。

CLI 默认仍使用 `defm_stackconv_depth`，以兼容现有 AS-Depth-2 checkpoint。通过
`--depth-model <model_id>` 选择其他模型；使用 `--depth-model auto` 时按 checkpoint envelope manifest
或 AS-Depth 历史路径 registry 解析。inverse-depth 模型的 raw depth 输入和模型输出都会在适配层显式
转换，交给 AnyGrasp 的结果始终是 finite、非负、`float32`、meter 深度。

### 环境

完整链路需要：

- Linux x86-64；
- AnyGrasp 提供 CPython 3.6–3.14 的版本化二进制，运行时按当前 ABI 自动选择；完整 AS-Depth
  链路建议继续使用 CPython 3.10，以匹配现有部署和依赖组合；
- CUDA、MinkowskiEngine、新版 AnyGrasp 许可证及其系统依赖；
- 新入口还需安装 `requirements-realsense.txt`；
- AS-Depth 和 AnyGrasp checkpoint 均通过外部路径或未跟踪的 `ckpts/` 提供。

安装 Python 依赖：

```bash
python3.10 -m pip install -r requirements.txt
python3.10 -m pip install -r requirements-asdepth.txt
python3.10 -m pip install -r requirements-realsense.txt
```

仓库不绑定环境管理器；上述 requirements 可用于标准 `venv`、Conda 或其他兼容环境。

完整抓取入口复用现有 `get_pose.py`，而该文件在模块加载时会导入 `pyrealsense2`，所以
`asdepth_pipeline.py` 的离线 RGB-D 模式同样需要上述 RealSense Python 依赖。仅深度预测可使用
下方独立入口绕过 AnyGrasp、RealSense 和 Piper。

### AnyGrasp 新许可证

2026-07 的新版 GSNet 不再支持仓库原有的 `lib_cxx.so` 和旧许可证。先在 Linux 目标机获取 feature ID：

```bash
python -c "from anygrasp_runtime import load_gsnet_module; print(load_gsnet_module().get_feature_id())"
```

按上游申请流程取得新许可证后，将整个目录解压到仓库根目录的 `license/`；该目录必须包含
`licenseCfg.json`，并已被 `.gitignore` 排除。可在目标机验证：

```bash
python -c "from anygrasp_runtime import load_gsnet_module; load_gsnet_module().check_license('license')"
```

上游接口和 steering 参数说明见 `USAGE.md`。该文件保持官方内容，其中提到的官方 demo 位于
`/Users/felix/Projects/anygrasp_sdk/grasp_detection`；本仓库的 `demo.py` 是项目场景入口。

### macOS 仅深度预测

macOS（包括 Apple Silicon）不能加载仓库中的 Linux x86-64 AnyGrasp GSNet 二进制。如果只想在
Mac 上验证任意 AS-Depth catalog 模型的 RGB-D 深度预测，可使用不导入 AnyGrasp、RealSense 或
Piper 的独立入口：

```bash
python asdepth_depth_only.py \
  --depth-checkpoint /path/to/model.ckpt \
  --depth-model defm_stackconv_depth \
  --rgb-image example_data/color.png \
  --depth-image example_data/depth.png \
  --save-dir debug/asdepth-only \
  --device mps
```

该入口只需要 `requirements-asdepth.txt` 中的依赖，输出 `pred_depth.npy` 和
`run_metadata.json`。它不能生成 AnyGrasp 抓取姿态；完整抓取仍需在 Linux x86-64 环境运行
`asdepth_pipeline.py`。

GSNet/AnyGrasp 二进制还受 CUDA、MinkowskiEngine、glibc 和许可证环境约束，必须在目标 Linux
服务器上做最终验证。macOS 可运行帮助、静态检查和轻量测试，但不能运行仓库内的 GSNet 二进制。

### 离线 RGB-D

```bash
python asdepth_pipeline.py \
  --depth-checkpoint /path/to/model.ckpt \
  --depth-model defm_stackconv_depth \
  --grasp-checkpoint ckpts/checkpoint-rs.tar \
  --rgb-image example_data/color.png \
  --depth-image example_data/depth.png \
  --save-dir debug/asdepth
```

### RealSense 采集

不传 `--rgb-image/--depth-image` 时，入口复用现有 `capture_one_frame()`：

```bash
python asdepth_pipeline.py \
  --depth-checkpoint /path/to/model.ckpt \
  --depth-model defm_stackconv_depth \
  --grasp-checkpoint ckpts/checkpoint-rs.tar \
  --save-dir debug/asdepth
```

默认只生成深度和抓取位姿，不控制机械臂。人工确认环境安全后，显式添加：

```bash
python asdepth_pipeline.py ... --execute-arm
```

每次运行输出：

```text
<run-dir>/
├── pred_depth.npy       # float32，meter，原相机尺寸
├── grasp_pose.txt
└── run_metadata.json
```

输入 raw depth 默认按毫米解析（`--depth-scale 1000`），`0` 和 `>=10 m` 为无效值。
默认预处理为 `input_size=518`、`lower_bound`、14 像素 patch 对齐，预测使用 nearest
恢复到原相机尺寸后再交给 AnyGrasp。

## 原有入口

- `pipline.py`：原远程单目深度、AprilTag、AnyGrasp、Piper 流程。
- `get_pose.py`：RealSense RGB-D 与 AnyGrasp。
- `grasp_piper.py`：抓取位姿转换和 Piper 控制。
- `demo.py`：保留本项目 D435 内参、`test_` 数据前缀和 workspace 的离线示例，通过新版 SDK 接口运行。
- `demo.sh`：本项目场景启动脚本；可通过 `CHECKPOINT_PATH`、`DATA_DIR` 或附加 CLI 参数覆盖默认值。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试不加载真实 AS-Depth checkpoint、不导入 GSNet，也不控制机械臂。真实 checkpoint、CUDA、
RealSense、AnyGrasp 和 Piper 属于 Linux 服务器/硬件验收。

## 第三方代码

`vendor/as_depth-0.3.0-py3-none-any.whl` 与旧 `asdepth_depth/` 兼容快照来自 AS-Depth，并包含
ByteDance Seed CDM、Meta DINOv2、ETH Zurich DeFM 和 MoGe 的派生实现。完整来源、范围和校验值见
`vendor/README.md`、`ASDEPTH_NOTICE` 与 `ASDEPTH_LICENSE`。这些内容与闭源 AnyGrasp SDK/许可证
相互独立。
