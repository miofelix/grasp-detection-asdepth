#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
import time

import numpy as np
import open3d as o3d
from piper_sdk import C_PiperInterface_V2

from camera_capture import capture_one_frame
from get_pose import run_anygrasp


def to_um(m):  # 米 → 微米（mm*1e3 == um）
    return int(round(m * 1e6))


def deg_to_mdeg(deg):  # 度 → 毫度
    return int(round(deg * 1e3))


def rad2deg(angles):
    return tuple(np.degrees(angles))


def euler_zyx(R):
    """
    欧拉顺序 ZYX（yaw-pitch-roll），对应 R = Rz * Ry * Rx
    返回 (rx, ry, rz) = (roll_x, pitch_y, yaw_z)，单位：弧度
    """
    # pitch = asin(-r31)
    pitch = math.asin(-R[2, 0])
    # roll  = atan2(r32, r33)
    roll = math.atan2(R[2, 1], R[2, 2])
    # yaw   = atan2(r21, r11)
    yaw = math.atan2(R[1, 0], R[0, 0])
    return roll, pitch, yaw


def euler_xyz(R):
    """
    欧拉顺序 XYZ（rx-ry-rz），对应 R = Rx * Ry * Rz
    返回 (rx, ry, rz)，单位：弧度
    """
    # ry = asin(-r13)
    ry = math.asin(-R[0, 2])
    # rx = atan2(r23, r33)
    rx = math.atan2(R[1, 2], R[2, 2])
    # rz = atan2(r12, r11)
    rz = math.atan2(R[0, 1], R[0, 0])
    return rx, ry, rz

    # ------- ZYX 欧拉角分解（R = Rz * Ry * Rx）-------


def clamp(x, lo=-1.0, hi=1.0):
    return min(max(x, lo), hi)


GRIPPER_DISTANCE = 0.07
T_cam2base = np.array(
    [
        [-0.03188415, -0.64642446, 0.7623115, -0.02826907],
        [-0.99742629, -0.02842686, -0.06582336, -0.2205848],
        [0.06421995, -0.76244825, -0.64385438, 0.57842954],
        [0.0, 0.0, 0.0, 1.0],
    ]
)


def visualize_pose_with_origin(T):
    t = T[:3, 3]
    R = T[:3, :3]

    # 基座原点坐标系 (大一点，代表世界坐标)
    origin_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.2)

    # 物体坐标系 (小一点，代表检测到的姿态)
    obj_axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=0.1)
    obj_axes.rotate(R, center=[0, 0, 0])
    obj_axes.translate(t)

    # 小球标记物体位置
    sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.015)
    sphere.paint_uniform_color([0.8, 0.8, 0.8])
    sphere.translate(t)

    # 一条线：从原点到物体位置
    line_points = [[0, 0, 0], t.tolist()]
    lines = [[0, 1]]
    colors = [[1, 0, 1]]  # 紫色
    line_set = o3d.geometry.LineSet(
        points=o3d.utility.Vector3dVector(line_points),
        lines=o3d.utility.Vector2iVector(lines),
    )
    line_set.colors = o3d.utility.Vector3dVector(colors)

    # 显示
    o3d.visualization.draw_geometries([origin_axes, obj_axes, sphere, line_set])


def move_with_check(piper, X, Y, Z, RX, RY, RZ, pos_tol=5_000, ang_tol=5_000, max_iter=1000):
    """
    控制机械臂移动并等待收敛
    - piper: 控制接口
    - (X,Y,Z): 目标位置 (μm)
    - (RX,RY,RZ): 目标姿态 (毫度)
    - pos_tol: 位置误差阈值 (μm)
    - ang_tol: 姿态误差阈值 (毫度)
    - max_iter: 循环最大次数，防止死循环
    """
    cnt = 0
    while True:
        cnt += 1
        piper.EndPoseCtrl(X, Y, Z, RX, RY, RZ)
        ep = piper.GetArmEndPoseMsgs().end_pose

        if (
            abs(ep.RX_axis - X) < pos_tol
            and abs(ep.RY_axis - Y) < pos_tol
            and abs(ep.RZ_axis - Z) < pos_tol
            and abs(ep.Rx - RX) < ang_tol
            and abs(ep.Ry - RY) < ang_tol
            and abs(ep.Rz - RZ) < ang_tol
        ):
            break
        if cnt > max_iter:
            print("❌ 到位超时，退出循环")
            break
        time.sleep(0.005)


def to_range_minus90_90(angle_deg: float) -> float:
    """
    输入: 任意角度 (度)
    输出: 在 [-90, 90] 内，与输入等价 (相差 180° 以内)
    """
    # 映射到 [-180,180)
    angle = (angle_deg + 180) % 360 - 180
    # 如果超出 [-90,90]，则加/减180翻转
    if angle > 90:
        angle -= 180
    elif angle < -90:
        angle += 180
    return angle


def run_pipeline(R_cam, t_cam, width):
    T_cam2obj = np.eye(4, dtype=float)
    T_cam2obj[:3, :3] = R_cam
    T_cam2obj[:3, 3] = t_cam
    # ===== 1) 可视化原始位姿坐标系=====
    R_base_orig = (T_cam2base @ T_cam2obj)[:3, :3]
    p_base_orig = (T_cam2base @ T_cam2obj)[:3, 3]
    rx0, ry0, rz0 = euler_zyx(R_base_orig)
    rx0_d, ry0_d, rz0_d = np.degrees([rx0, ry0, rz0])
    print("原始位姿 x y z (m):", *[f"{v:.8f}" for v in p_base_orig])
    print("原始位姿 rx ry rz (deg):", *[f"{v:.8f}" for v in (rx0_d, ry0_d, rz0_d)])

    # ===== 2) 轴系映射（在物体局部做重排：右乘）=====
    # 目标：new_x = old_z, new_y = old_y, new_z = old_x
    M = np.eye(4, dtype=float)
    M[:3, :3] = np.array(
        [
            [0, 0, 1],  # new_x ← old_z
            [0, -1, 0],  # new_y ← -old_y
            [1, 0, 0],  # new_z ← old_x
        ],
        dtype=float,
    )
    T_cam2obj = T_cam2obj @ M

    # ===== 3) 在物体局部 z 轴方向后退 10 cm（右乘）=====
    distance = GRIPPER_DISTANCE  # 10 cm
    delta = np.eye(4, dtype=float)
    delta[:3, 3] = np.array([0, 0, -distance], dtype=float)
    T_cam2obj = T_cam2obj @ delta

    distance_mid = 0.15
    delta_mid = np.eye(4, dtype=float)
    delta_mid[:3, 3] = np.array([0, 0, -distance_mid], dtype=float)
    T_cam2obj_mid = T_cam2obj @ delta_mid
    # ===== 4) 左乘到基座坐标系 =====
    T_base2obj = T_cam2base @ T_cam2obj
    T_base2obj_mid = T_cam2base @ T_cam2obj_mid
    # ===== 5) 提取基座下的姿态与位置 =====
    R_base = T_base2obj[:3, :3].copy()
    p_base = T_base2obj[:3, 3].copy()  # (x,y,z) in meters
    p_mid = T_base2obj_mid[:3, 3].copy()
    # visualize_pose_with_origin(T_base2obj)

    rx, ry, rz = euler_zyx(R_base)
    rx_d, ry_d, rz_d = np.degrees([rx, ry, rz])

    print("退回10cm后 x y z (m):", *[f"{v:.8f}" for v in p_base])
    print("退回10cm后 rx ry rz (deg):", *[f"{v:.8f}" for v in (rx_d, ry_d, rz_d)])
    print("退回10cm后再退15cm x y z (m):", *[f"{v:.8f}" for v in p_mid])
    X_mid, Y_mid, Z_mid = map(to_um, p_mid)

    X, Y, Z = map(to_um, p_base)

    # 2) SDK期望：位置=mm*1e3(即 μm)，角度=deg*1e3(毫度)
    RX, RY, RZ = (
        deg_to_mdeg(to_range_minus90_90(rx0_d)),
        deg_to_mdeg(to_range_minus90_90(ry0_d) + 85),
        deg_to_mdeg(to_range_minus90_90(rz0_d)),
    )

    print("Target (um, mdeg):", X, Y, Z, RX, RY, RZ)

    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    while not piper.EnablePiper():
        time.sleep(0.01)
    piper.GripperCtrl(0, 1000, 0x02, 0)
    piper.GripperCtrl(0, 1000, 0x01, 0)
    # 1)张开两只夹爪（如有）
    range = 90
    count = 0
    while True:
        count += 1
        piper.GripperCtrl(abs(round(range * 1000)), 1000, 0x01, 0)
        g = piper.GetArmGripperMsgs().gripper_state.grippers_angle
        if g > 80000:
            break
        time.sleep(0.005)
    print("已张开夹爪")
    # 运动参数
    piper.MotionCtrl_2(0x01, 0x00, 15, 0x00)  # 速度等
    piper.EndPoseCtrl(15_000, 0, 275_000, 0, 85_000, 0)
    print("夹爪已运动到零点上方")
    time.sleep(1.0)

    # 2) 移动到水杯上方 15 cm
    target_Z = Z + int(100e3)
    move_with_check(piper, X_mid, Y_mid, Z_mid, RX, RY, RZ)
    time.sleep(1.0)
    print("已移动到水杯后方 15 cm")

    # 3) 下到水杯位置
    move_with_check(piper, X, Y, Z, RX, RY, RZ)
    time.sleep(1.0)
    print("已下到水杯位置")

    # 4) 收拢夹爪（抓水杯）
    grip_range = int(width * 1e6) * 0.5
    range = 0
    count = 0
    while True:
        count += 1
        # print(piper.GetArmGripperMsgs())
        if count == 500:
            range = 500
            count = 0
        elif count == 100:
            range = 0
        piper.GripperCtrl(abs(round(grip_range)), 1000, 0x01, 0)
        g = piper.GetArmGripperMsgs().gripper_state.grippers_angle
        if g > 80000:
            break
        time.sleep(0.005)
    print("已收拢夹爪")
    time.sleep(2.0)

    # 5) 抬起水杯
    target_Z = Z + int(100e3)
    move_with_check(piper, X, Y, target_Z, RX, RY, RZ)
    print("✅ 已成功抓起水杯")

    # move_with_check(piper, 520_000, -460_000, 310_000, 60_000, 90_000, 0)
    # range=0
    # count=0
    # while True:
    #    count+=1
    #    #print(piper.GetArmGripperMsgs())
    #    if count==500:
    #        range=500
    #        count=0
    #    elif count==100:
    #        range=0
    #    piper.GripperCtrl(abs(round(100_000)), 1000, 0x01, 0)
    #    g = piper.GetArmGripperMsgs().gripper_state.grippers_angle
    #    if g > 80000:
    #        break
    #    time.sleep(0.005)
    # print("已收拢夹爪")
    # time.sleep(2.0)
    # piper.JointCtrl(0, 0, 0, 0, 0, 0)
    # piper.GripperCtrl(abs(0), 1000, 0x01, 0)


if __name__ == "__main__":
    # ==== 创建实验保存目录 ====
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
    parser.add_argument(
        "--top_down_grasp", default=True, action="store_true", help="Output top-down grasps."
    )
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
    capture = capture_one_frame(cfgs.save_dir, backend=cfgs.camera_backend)
    cfgs.depth_scale = capture.raw_units_per_meter
    cfgs.camera_intrinsics = {
        "fx": capture.intrinsics.fx,
        "fy": capture.intrinsics.fy,
        "cx": capture.intrinsics.cx,
        "cy": capture.intrinsics.cy,
    }
    save_dir = str(capture.run_dir)
    R_cam, t_cam, width = run_anygrasp(save_dir, cfgs, data_dir=save_dir)
    run_pipeline(R_cam, t_cam, width)
