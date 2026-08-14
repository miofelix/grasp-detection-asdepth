import argparse
import os

import cv2
import numpy as np
import open3d as o3d
from PIL import Image

from anygrasp_runtime import create_detector, predict_grasps
from camera_capture import capture_one_frame as capture_rgbd_frame

# ----------------- 参数 -----------------


# ----------------- RGB-D 相机获取一帧 -----------------
def capture_one_frame(base_dir, camera_backend="orbbec"):
    """兼容旧调用方式，默认使用 Orbbec，返回采集目录字符串。"""

    return str(capture_rgbd_frame(base_dir, backend=camera_backend).run_dir)


# ----------------- AnyGrasp Demo -----------------
def run_anygrasp(save_dir, cfgs, data_dir=None, rgb=None, depth=None):
    detector = create_detector(cfgs)
    # 读取图像
    colors = None
    depths = None
    points_z = None
    if data_dir is not None:
        colors = np.array(Image.open(os.path.join(data_dir, "color.png")), dtype=np.float32) / 255.0
        depths = np.array(Image.open(os.path.join(data_dir, "depth.png")))
        points_z = depths / float(getattr(cfgs, "depth_scale", 1000.0))
    else:
        colors = rgb
        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)  # 转成 RGB
        colors = colors.astype(np.float32) / 255.0
        depths = depth
        points_z = depths
    # 在线模式由相机 SDK 提供对齐后彩色相机内参；离线旧调用保留历史默认值。
    camera_intrinsics = getattr(cfgs, "camera_intrinsics", None) or {
        "fx": 616.22601724,
        "fy": 615.78839082,
        "cx": 315.33494299,
        "cy": 251.59150012,
    }
    fx = float(camera_intrinsics["fx"])
    fy = float(camera_intrinsics["fy"])
    cx = float(camera_intrinsics["cx"])
    cy = float(camera_intrinsics["cy"])
    xmin, xmax = -0.10, 0.10
    ymin, ymax = -0.2, 0.07
    zmin, zmax = 0.2, 1.0
    lims = [xmin, xmax, ymin, ymax, zmin, zmax]

    # 点云计算
    xmap, ymap = np.meshgrid(np.arange(depths.shape[1]), np.arange(depths.shape[0]))

    points_x = (xmap - cx) / fx * points_z
    points_y = (ymap - cy) / fy * points_z

    mask = (points_z > 0) & (points_z < 1)
    points = np.stack([points_x, points_y, points_z], axis=-1)[mask].astype(np.float32)
    colors = colors[mask].astype(np.float32)

    if len(points) == 0:
        raise RuntimeError("No valid point remains after depth filtering")
    print("Point cloud range:", points.min(axis=0), points.max(axis=0))

    # 推理抓取点
    gg = predict_grasps(
        detector,
        points,
        lims,
        top_down_grasp=bool(getattr(cfgs, "top_down_grasp", False)),
        dense_grasp=False,
        collision_detection=True,
    )

    if gg is None:
        print("No Grasp detected after collision detection!")
        return

    gg_pick = gg[0:20]

    print("Top grasp scores:", gg_pick.scores)
    print("得分最高的物体位姿：", gg[0])
    print("最高分 grasp score:", gg_pick[0].score)

    # 可视化
    if cfgs.debug:
        cloud = o3d.geometry.PointCloud()
        cloud.points = o3d.utility.Vector3dVector(points)
        cloud.colors = o3d.utility.Vector3dVector(colors)
        trans_mat = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]])
        cloud.transform(trans_mat)
        grippers = gg.to_open3d_geometry_list()
        cloud_path = os.path.join(save_dir, "scene_cloud_14b.ply")
        o3d.io.write_point_cloud(cloud_path, cloud)

        for gripper in grippers:
            gripper.transform(trans_mat)
        o3d.visualization.draw_geometries([*grippers, cloud])
        o3d.visualization.draw_geometries([grippers[0], cloud])
    return gg[0].rotation_matrix, gg[0].translation, gg[0].width


# ----------------- 主程序 -----------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--checkpoint_path",
        default="ckpts/checkpoint_detection.tar",
        help="AnyGrasp 2026 detection checkpoint path",
    )
    parser.add_argument(
        "--max_gripper_width", type=float, default=0.1, help="Maximum gripper width (<=0.1m)"
    )
    parser.add_argument("--gripper_height", type=float, default=0.03, help="Gripper height")
    parser.add_argument("--top_down_grasp", action="store_true", help="Output top-down grasps.")
    parser.add_argument("--debug", default=True, action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--save_dir", default="debug/funny", help="Directory to save captured frame"
    )
    parser.add_argument(
        "--camera_backend",
        choices=["orbbec", "realsense", "auto"],
        default="orbbec",
    )
    cfgs = parser.parse_args()
    cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
    capture = capture_rgbd_frame(cfgs.save_dir, backend=cfgs.camera_backend)
    cfgs.depth_scale = capture.raw_units_per_meter
    cfgs.camera_intrinsics = {
        "fx": capture.intrinsics.fx,
        "fy": capture.intrinsics.fy,
        "cx": capture.intrinsics.cx,
        "cy": capture.intrinsics.cy,
    }
    run_anygrasp(str(capture.run_dir), cfgs, data_dir=str(capture.run_dir))
