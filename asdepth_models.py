"""列出当前项目可用于深度预测和抓取流水线的 AS-Depth 模型。"""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="列出 AS-Depth canonical 模型 catalog")
    parser.add_argument("--json", action="store_true", help="输出机器可读 JSON")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    from asdepth_depth import list_depth_models

    models = list_depth_models()
    if args.json:
        print(
            json.dumps(
                [
                    {
                        "model_id": item.model_id,
                        "model_version": item.model_version,
                        "config_hash": item.config_hash,
                        "entrypoint": item.entrypoint,
                        "native_depth": item.native_depth,
                        "sparse_raw_depth": item.sparse_raw_depth,
                    }
                    for item in models
                ],
                ensure_ascii=False,
                indent=2,
                sort_keys=True,
            )
        )
        return 0

    print(f"{'MODEL_ID':48} {'NATIVE_DEPTH':15} ENTRYPOINT")
    for item in models:
        print(f"{item.model_id:48} {item.native_depth:15} {item.entrypoint}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
