# grasp_detection

本仓库保留师兄原有的 AnyGrasp、点云和 Piper 控制代码，同时新增了一条本地 AS-Depth-2
RGB-D 推理入口。旧的 `pipline.py` 未修改，仍对应原来的远程 GAVP + AprilTag 流程。

## AS-Depth 抓取入口

新入口的数据流为：

```text
RealSense 对齐 RGB-D
  → AS-Depth-2 defm_stackconv_depth
  → meter 深度图
  → 现有 get_pose.run_anygrasp
  → 可选 grasp_piper.run_pipline
```

只迁入了 `defm_stackconv_depth` 的模型数学代码、必要的 DINOv2/DeFM/MoGe 派生实现、
checkpoint loader 和 RGB-D 预处理。训练、评估、数据集、Web、TensorRT、其他模型和权重均未迁移。

### 环境

完整链路需要：

- Linux x86-64；
- CPython 3.10，对应仓库中的
  `gsnet_versions/gsnet.cpython-310-x86_64-linux-gnu.so`；
- CUDA、AnyGrasp 许可证及其系统依赖；
- 新入口还需安装 `requirements-realsense.txt`；
- AS-Depth 和 AnyGrasp checkpoint 均通过外部路径或未跟踪的 `ckpts/` 提供。

安装 Python 依赖：

```bash
python3.10 -m pip install -r requirements.txt
python3.10 -m pip install -r requirements-asdepth.txt
python3.10 -m pip install -r requirements-realsense.txt
```

新增入口复用现有 `get_pose.py`，而该文件在模块加载时会导入 `pyrealsense2`，所以当前离线 RGB-D
模式同样需要上述 RealSense Python 依赖。为保持最小迁移，没有改动师兄的导入结构。

现有 GSNet/AnyGrasp 二进制还可能受 CUDA、libstdc++、OpenSSL 1.1 和许可证环境约束，必须在目标
Linux 服务器上做最终验证。macOS 可运行帮助、静态检查和轻量测试，但不能运行仓库内的 GSNet 二进制。

### 离线 RGB-D

```bash
python asdepth_pipeline.py \
  --depth-checkpoint /path/to/asdepth2.ckpt \
  --grasp-checkpoint ckpts/checkpoint-rs.tar \
  --rgb-image example_data/color.png \
  --depth-image example_data/depth.png \
  --save-dir debug/asdepth
```

### RealSense 采集

不传 `--rgb-image/--depth-image` 时，入口复用现有 `capture_one_frame()`：

```bash
python asdepth_pipeline.py \
  --depth-checkpoint /path/to/asdepth2.ckpt \
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
- `demo.py`：原 AnyGrasp 离线示例。

## 测试

```bash
python -m unittest discover -s tests -v
```

测试不加载真实 AS-Depth checkpoint、不导入 GSNet，也不控制机械臂。真实 checkpoint、CUDA、
RealSense、AnyGrasp 和 Piper 属于 Linux 服务器/硬件验收。

## 第三方代码

`asdepth_depth/` 迁移自 AS-Depth，并包含 ByteDance Seed CDM、Meta DINOv2、ETH Zurich DeFM
和 MoGe 的派生实现。各 vendor 文件保留原始版权声明；完整说明见 `ASDEPTH_NOTICE` 和
`ASDEPTH_LICENSE`。这些文件与仓库已有闭源 AnyGrasp SDK/许可证相互独立。
