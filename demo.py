"""使用本项目场景参数调用 AnyGrasp 2026 SDK 的离线示例。"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import open3d as o3d
from PIL import Image

from anygrasp_runtime import create_detector, predict_grasps

# 本项目 D435 场景配置；官方 SDK 代码保持在上游仓库中，不在这里修改。
CAMERA_INTRINSICS = (488.28772, 488.28772, 315.879547, 213.037033)
WORKSPACE_LIMITS = (-0.24, 0.40, -0.4, 0.4, 0.0, 3.0)
DEPTH_SCALE = 1000.0
DEPTH_TRUNCATION_M = 1.0
DEFAULT_CHECKPOINT_PATH = "ckpts/checkpoint_detection.tar"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="使用本项目 D435 场景配置运行 AnyGrasp 离线检测",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--checkpoint_path",
        default=DEFAULT_CHECKPOINT_PATH,
        help="AnyGrasp 2026 detection checkpoint path",
    )
    parser.add_argument("--data-dir", default="example_data", help="RGB-D 数据目录")
    parser.add_argument("--image-prefix", default="test_", help="color/depth 文件名前缀")
    parser.add_argument("--max_gripper_width", type=float, default=0.1)
    parser.add_argument("--gripper_height", type=float, default=0.03)
    parser.add_argument("--top_down_grasp", action="store_true")
    parser.add_argument("--debug", action="store_true", help="显示 Open3D 可视化")
    return parser


def load_point_cloud(data_dir: str | Path, prefix: str) -> tuple[np.ndarray, np.ndarray]:
    directory = Path(data_dir).expanduser().resolve()
    color_path = directory / f"{prefix}color.png"
    depth_path = directory / f"{prefix}depth.png"
    if not color_path.is_file() or not depth_path.is_file():
        raise FileNotFoundError(f"missing RGB-D input: {color_path}, {depth_path}")

    colors = np.asarray(Image.open(color_path).convert("RGB"), dtype=np.float32) / 255.0
    depths = np.asarray(Image.open(depth_path))
    if depths.ndim != 2 or depths.shape != colors.shape[:2]:
        raise ValueError(f"invalid RGB-D shapes: color={colors.shape}, depth={depths.shape}")

    fx, fy, cx, cy = CAMERA_INTRINSICS
    xmap, ymap = np.meshgrid(np.arange(depths.shape[1]), np.arange(depths.shape[0]))
    points_z = depths / DEPTH_SCALE
    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    valid = (points_z > 0) & (points_z < DEPTH_TRUNCATION_M)
    points = np.stack([points_x, points_y, points_z], axis=-1)[valid].astype(np.float32)
    colors = colors[valid].astype(np.float32)
    if len(points) == 0:
        raise RuntimeError("No valid point remains after depth filtering")
    return points, colors


def render_results(points: np.ndarray, colors: np.ndarray, grasps: object) -> None:
    cloud = o3d.geometry.PointCloud()
    cloud.points = o3d.utility.Vector3dVector(points)
    cloud.colors = o3d.utility.Vector3dVector(colors)
    transform = np.array(
        [[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]],
        dtype=np.float64,
    )
    cloud.transform(transform)
    grippers = grasps.to_open3d_geometry_list()
    for gripper in grippers:
        gripper.transform(transform)
    o3d.visualization.draw_geometries([*grippers, cloud])
    o3d.visualization.draw_geometries([grippers[0], cloud])


def run(args: argparse.Namespace) -> object | None:
    args.max_gripper_width = max(0.0, min(0.1, args.max_gripper_width))
    detector = create_detector(args)
    points, colors = load_point_cloud(args.data_dir, args.image_prefix)
    print("Point cloud range:", points.min(axis=0), points.max(axis=0))

    grasps = predict_grasps(
        detector,
        points,
        WORKSPACE_LIMITS,
        top_down_grasp=bool(args.top_down_grasp),
        dense_grasp=False,
        collision_detection=True,
    )
    if grasps is None:
        print("No Grasp detected after collision detection!")
        return None

    top_grasps = grasps[:20]
    print("Top grasp scores:", top_grasps.scores)
    print("得分最高的物体的位姿：", grasps[0])
    print("grasp score:", top_grasps[0].score)
    if args.debug:
        render_results(points, colors, grasps)
    return grasps[0]


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return 0 if run(args) is not None else 1
    except (FileNotFoundError, RuntimeError, ValueError) as exc:
        print(f"error: {exc}")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
