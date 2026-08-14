#!/usr/bin/env bash

# 安装可从 PyPI 获取的依赖。可直接使用当前环境，也可创建标准 venv；Conda 环境同样适用。
# 用法: bash install.sh          # 使用当前 python3
#       bash install.sh --venv   # 在项目目录创建 .venv

set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

USE_VENV=0
for arg in "$@"; do
  if [[ "$arg" == "--venv" ]]; then USE_VENV=1; fi
done

if ! command -v python3 >/dev/null 2>&1; then
  echo "错误: 未找到 python3，请先安装 Python 3.10+。" >&2
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
echo "========== Python 依赖已安装 =========="
echo "AnyGrasp 官方 GSNet 二进制已放在 gsnet_versions/，运行时按当前 CPython ABI 自动选择。"
echo "请把新版 AnyGrasp 许可证解压为 $ROOT/license/（必须包含 licenseCfg.json）。"
echo "AS-Depth 依赖: python3 -m pip install -r requirements-asdepth.txt"
echo "RealSense 依赖: python3 -m pip install -r requirements-realsense.txt"
