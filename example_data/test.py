import cv2
import numpy as np
import open3d as o3d

def main():
    # 读取彩色图
    color_raw = o3d.io.read_image("test_color.png")

    # 用 OpenCV 读取深度图并 resize
    depth_np = cv2.imread("test_depth.png", cv2.IMREAD_UNCHANGED)  # 保持16位深度
    depth_resized = cv2.resize(depth_np, (640, 480), interpolation=cv2.INTER_LINEAR)

    # 转换为 Open3D 的 Image
    depth_raw = o3d.geometry.Image(depth_resized)

    # 设置深度图参数
    depth_scale = 1000.0   # mm → m
    depth_trunc = 2.0      # 截断3米以内的点

    # 生成RGBD图像
    rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(
        color_raw,
        depth_raw,
        depth_scale=depth_scale,
        depth_trunc=depth_trunc,
        convert_rgb_to_intensity=False
    )

    print("RGBD image created.")

    # 设置相机内参 (你需要根据相机实际参数修改)
    fx, fy = 488.28772, 488.28772
    cx, cy = 315.879547, 213.037033
    intrinsic = o3d.camera.PinholeCameraIntrinsic(
        width=640,
        height=480,
        fx=fx, fy=fy, cx=cx, cy=cy
    )

    # 生成点云
    pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
        rgbd_image,
        intrinsic
    )

    # 翻转到常见坐标系
    pcd.transform([[1, 0, 0, 0],
                   [0, -1, 0, 0],
                   [0, 0, -1, 0],
                   [0, 0, 0, 1]])

    # 可视化
    o3d.visualization.draw_geometries([pcd])

if __name__ == "__main__":
    main()
