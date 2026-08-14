# AS-Depth 多模型推理说明

当前项目通过内置的 `as-depth 0.3.0` wheel 使用 AS-Depth Research 的 canonical model catalog。
模型身份由 `model_id + model_version + config_hash` 组成，checkpoint 不再固定绑定
`defm_stackconv_depth`。

## 模型选择

```bash
python asdepth_models.py
python asdepth_models.py --json
```

深度-only 与完整抓取入口均接受：

```text
--depth-checkpoint <checkpoint 文件、envelope 目录或 Accelerate 目录>
--depth-model <canonical model_id | auto>
```

默认值为 `defm_stackconv_depth`，用于兼容已有 AS-Depth-2 部署。`auto` 的解析优先级为：

```text
checkpoint manifest → 历史 checkpoint 路径 registry → 报错
```

普通单文件 checkpoint 若路径不在历史 registry 中，应显式传 `--depth-model`。

## 活跃 catalog

- CDM / 双 DINOv2：`cdm_d435_pretrained_disp`、`cdm_l515_pretrained_disp`、
  `cdm_vitl_pretrained_disp`、`cdm_vitb_pretrained_disp`、`cdm_vitl_pretrained_depth`、
  `cdm_vitl_depth`、`cdm_vitl_disp`。
- DeFM ViT-L14：`defm_vit_l14_depth`、`defm_vit_l14_depth_runner`、
  `defm_vit_l14_random_raw`、`defm_vit_l14`、`fanyu_defm`。
- DINOv2 stack-conv：`stackconv_depth_random_raw`、`stackconv_depth_multidataset`、
  `stackconv_depth_multidataset_new_fusion`。
- DeFM stack-conv：`defm_stackconv_no_adaptor_depth`、`defm_stackconv_depth`、
  `defm_stackconv_depth_sparse_raw`。
- Auxiliary/multitask：`defm_stackconv_mask_normal_depth`、
  `defm_stackconv_multitask_depth`。

## 深度语义

- metric-depth 模型接收米制 raw depth，并直接产生米制深度。
- inverse-depth 模型接收 inverse raw depth；适配层从相机米制深度安全转换，并把预测再转回 meter。
- sparse-raw 模型根据 catalog metadata 自动启用稀疏化预处理。
- multitask 模型仍以第一个输出作为深度；mask/normal 等辅助输出不进入 AnyGrasp。
- 最终输出统一为与相机帧同尺寸的二维 `float32`，单位 meter，invalid 固定为 `0.0`。

## Checkpoint 格式

支持 AS-Depth 正式 envelope、Accelerate checkpoint 目录，以及历史 `.ckpt`、`.pt`、`.bin`、
`.safetensors` 单文件。默认严格加载模型参数；pickle checkpoint 默认使用 `weights_only=True`，只有
显式添加 `--trusted-depth-checkpoint` 才允许读取受信任的旧自定义对象。

模型结构可在 macOS/CPU 上进行轻量检查；真实 checkpoint 的显存占用、速度和数值结果仍需在目标
CUDA 服务器上验证。
