#!/usr/bin/env bash
###
 # @Author: daniel
 # @Date: 2026-04-28 17:58:21
 # @LastEditTime: 2026-04-28 20:37:26
 # @LastEditors: daniel
 # @Description: 
 # @FilePath: /grasp_detection/install.sh
 # have a nice day
### 
# 安装 grasp_detection 流水线在 PyPI 上可获得的 Python 依赖。
# 用法: bash install.sh          # 使用当前 python3/pip3
#       bash install.sh --venv   # 在项目目录创建 .venv 并安装到其中


# conda create -n grasp_detection python=3.10 -y
# conda activate grasp_detection

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

USE_VENV=0
for arg in "$@"; do
  if [[ "$arg" == "--venv" ]]; then USE_VENV=1; fi
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "错误: 未找到 python3，请先安装 Python 3.8+。" >&2
  exit 1
fi

if [[ "$USE_VENV" -eq 1 ]]; then
  python3 -m venv "$ROOT/.venv"
  # shellcheck source=/dev/null
  source "$ROOT/.venv/bin/activate"
  echo "已激活虚拟环境: $ROOT/.venv"
fi

python3 -m pip install --upgrade pip setuptools wheel
python3 -m pip install -r "$ROOT/requirements.txt"

echo ""
echo "========== pip 依赖已安装 =========="
echo ""
echo "以下模块无法通过本 requirements 自动安装，请按你的环境单独配置:"
echo "  1) gsnet / AnyGrasp — 来自 AnyGrasp 官方 SDK，需将 SDK 中的 Python 包加入 PYTHONPATH"
echo "     或 pip install -e <anygrasp_sdk 中的 gsnet 目录>"
echo "  2) 深度服务客户端 — pipline 会调用 GAVP/http_client/depth_estimator.py，请保证该仓库/路径存在"
echo "     且其依赖已安装。"
echo "  3) pyrealsense2 — 用 RealSense 时在 Linux 等环境执行: pip install -r requirements-realsense.txt"
echo "     Mac 上 PyPI 常无此包，若不用相机采图可不管；用图可从文件读 depth，不必装 pyrealsense2。"
echo ""
echo "验证示例: python3 -c \"import cv2, apriltag, numpy, scipy; import open3d; import torch; print('ok')\""


pip install torch torchvision
pip install open3d
