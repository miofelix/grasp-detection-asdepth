import argparse
import datetime
import os

import cv2
import numpy as np
import open3d as o3d
import pyrealsense2 as rs
from PIL import Image

from anygrasp_runtime import create_detector, predict_grasps

# ----------------- 参数 -----------------


# ----------------- Realsense 获取一帧 -----------------
def capture_one_frame(base_dir):
    # === 带时间戳的子目录 ===
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    save_dir = os.path.join(base_dir, f"capture_{timestamp}")
    os.makedirs(save_dir, exist_ok=True)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)

    profile = pipeline.start(config)
    align = rs.align(rs.stream.color)

    depth_sensor = profile.get_device().first_depth_sensor()
    depth_scale = depth_sensor.get_depth_scale()
    print("Depth Scale is:", depth_scale)

    frame_count = 0
    color_path = os.path.join(save_dir, "color.png")
    depth_path = os.path.join(save_dir, "depth.png")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            aligned_frames = align.process(frames)
            aligned_depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()

            if not aligned_depth_frame or not color_frame:
                continue

            depth_image = np.asanyarray(aligned_depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())

            frame_count += 1
            if frame_count > 30:  # 等待相机稳定
                cv2.imwrite(color_path, color_image)
                cv2.imwrite(depth_path, depth_image)
                print(f"Saved one frame: {color_path}, {depth_path}")
                break
    finally:
        pipeline.stop()

    return save_dir


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
        points_z = depths / 1000.0
    else:
        colors = rgb
        colors = cv2.cvtColor(colors, cv2.COLOR_BGR2RGB)  # 转成 RGB
        colors = colors.astype(np.float32) / 255.0
        depths = depth
        points_z = depths
    # 相机内参（要改成你的相机参数）
    fx, fy = 616.22601724, 615.78839082
    cx, cy = 315.33494299, 251.59150012
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
    parser.add_argument("--checkpoint_path", required=True, help="Model checkpoint path")
    parser.add_argument(
        "--max_gripper_width", type=float, default=0.1, help="Maximum gripper width (<=0.1m)"
    )
    parser.add_argument("--gripper_height", type=float, default=0.03, help="Gripper height")
    parser.add_argument("--top_down_grasp", action="store_true", help="Output top-down grasps.")
    parser.add_argument("--debug", default=True, action="store_true", help="Enable debug mode")
    parser.add_argument(
        "--save_dir", default="debug/funny", help="Directory to save captured frame"
    )
    cfgs = parser.parse_args()
    cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))
    save_dir = capture_one_frame(cfgs.save_dir)
    run_anygrasp(save_dir, cfgs)
