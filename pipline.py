import os
import cv2
import apriltag
import numpy as np
import subprocess
import logging
from datetime import datetime
from get_pose import run_anygrasp,capture_one_frame
import argparse
from grasp_piper import run_pipline
parser = argparse.ArgumentParser()
parser.add_argument('--checkpoint_path', default="/home/wqz/anygrasp_sdk/grasp_detection/log/checkpoint_detection.tar",help='Model checkpoint path')
parser.add_argument('--max_gripper_width', type=float, default=0.1, help='Maximum gripper width (<=0.1m)')
parser.add_argument('--gripper_height', type=float, default=0.03, help='Gripper height')
parser.add_argument('--top_down_grasp', default=True,action='store_true', help='Output top-down grasps.')
parser.add_argument('--debug', default=True,action='store_true', help='Enable debug mode')
parser.add_argument('--save_dir', default='debug/funny', help='Directory to save captured frame')
cfgs = parser.parse_args()
cfgs.max_gripper_width = max(0, min(0.1, cfgs.max_gripper_width))

def depth2disparity(depth, return_mask=False):
    disparity = np.zeros_like(depth)
    non_negtive_mask = depth > 0
    disparity[non_negtive_mask] = 1.0 / depth[non_negtive_mask]
    if return_mask:
        return disparity, non_negtive_mask
    else:
        return disparity

def disparity2depth(disparity, **kwargs):
    return depth2disparity(disparity, **kwargs)

class AlgoValPlat():
    def __init__(self, cam_params):
        self.cam_params = cam_params
        self.tag_size = 0.08
        
        # 配置logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f'gavp_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
            ]
        )
        self.logger = logging.getLogger(__name__)
        self.logger.info(f"初始化AlgoValPlat，相机参数: {cam_params}, AprilTag尺寸: {self.tag_size}m")

    def revover_depth(self, rgb_image, relative_depth_map):
        self.logger.info("开始深度恢复过程")
        self.logger.debug(f"输入图像尺寸: {rgb_image.shape}, 相对深度图形状: {relative_depth_map.shape}")
        
        camera_matrix = np.array([[self.cam_params[0], 0, self.cam_params[2]],[0, self.cam_params[1], self.cam_params[3]],[0, 0, 1]])
        H_d, W_d = relative_depth_map.shape[:2]
        self.logger.debug(f"相机内参矩阵: \n{camera_matrix}")

        # 2. 转为灰度图并检测 AprilTag
        self.logger.info("开始AprilTag检测")
        gray = cv2.cvtColor(rgb_image, cv2.COLOR_BGR2GRAY)
        detector = apriltag.Detector()
        detections = detector.detect(gray)
        self.logger.info(f"检测到 {len(detections)} 个AprilTag")

        if len(detections) == 0:
            self.logger.error("未检测到 AprilTag，请检查图像")
            raise ValueError("未检测到 AprilTag，请检查图像。")

        scale_factors = []
        for i, det in enumerate(detections):
            self.logger.info(f"处理第 {i+1}/{len(detections)} 个AprilTag")
            corners = det.corners.astype(int)
            self.logger.debug(f"AprilTag {i+1} 角点坐标: {corners}")
            
            tag2cam, e1, e2 = detector.detection_pose(det, self.cam_params)
            tag2cam[:3,3:] *= self.tag_size
            image_points = det.corners  # 检测到的 2D 角点
            self.logger.debug(f"AprilTag {i+1} 位姿矩阵: \n{tag2cam}")

            # 3. 计算 AprilTag 中心在图像中的坐标
            center_uv = np.mean(corners, axis=0).astype(int)  # (u, v)
            u, v = int(center_uv[0]), int(center_uv[1])
            self.logger.debug(f"AprilTag {i+1} 中心坐标: ({u}, {v})")

            # 4. 获取相对深度图中对应位置的深度值（插值或邻近）
            if v >= H_d or u >= W_d or u < 0 or v < 0:
                self.logger.warning(f"AprilTag {i+1} 中心坐标超出深度图范围，跳过")
                continue

            relative_depth_tag = relative_depth_map[v, u]
            if relative_depth_tag <= 0:
                self.logger.warning(f"AprilTag {i+1} 中心位置相对深度值无效: {relative_depth_tag}，跳过")
                continue

            # 6. 计算尺度因子
            half_size = self.tag_size / 2
            object_points = np.array([[-half_size, -half_size, 0], [half_size, -half_size, 0], [half_size, half_size, 0], [-half_size, half_size, 0]])
            # 计算 AprilTag 在图像中的掩模
            relative_depth_map = relative_depth_map.squeeze(-1) if len(relative_depth_map.shape) == 3 else relative_depth_map
            mask = np.zeros_like(relative_depth_map)
            h, w = mask.shape
            cv2.fillConvexPoly(mask, image_points.round().astype(int), 1)

            # object_points: (4, 3), 假设为 [[0,0,0], [size,0,0], [size,size,0], [0,size,0]]
            # tvec: 平移向量 (3,)
            # rvec: 旋转向量 (3,)
            rotation_matrix = tag2cam[:3,:3]
            tvec = tag2cam[:3,3:]
            corner_3d_camera = (rotation_matrix @ object_points.T).T + tvec.ravel()  # (4, 3)

            # 2. 拟合 AprilTag 所在平面：Ax + By + Cz + D = 0
            # 使用三个点构造两个向量，计算法向量
            v1 = corner_3d_camera[1] - corner_3d_camera[0]
            v2 = corner_3d_camera[3] - corner_3d_camera[0]
            normal = np.cross(v1, v2)
            normal = normal / np.linalg.norm(normal)
            A, B, C = normal
            D = -np.dot(normal, corner_3d_camera[0])

            # 平面方程：Ax + By + Cz + D = 0  =>  z = -(Ax + By + D) / C （当 C ≠ 0）

            # 3. 获取 AprilTag 区域内的所有像素坐标
            y_coords, x_coords = np.where(mask > 0)  # (N,), (N,)

            # 4. 将这些像素反投影到相机光线方向（使用内参）
            # 假设 intrinsic_matrix 已定义：[[fx, 0, cx], [0, fy, cy], [0, 0, 1]]
            fx, fy, cx, cy = camera_matrix[0,0], camera_matrix[1,1], camera_matrix[0,2], camera_matrix[1,2]

            # 归一化平面坐标（去畸变后）
            x_norm = (x_coords - cx) / fx
            y_norm = (y_coords - cy) / fy

            # 相机光线方向向量（未归一化）
            ray_dirs = np.stack([x_norm, y_norm, np.ones_like(x_coords)], axis=1)  # (N, 3)

            # 5. 求光线与 AprilTag 平面的交点
            # 光线：P(t) = t * ray_dir
            # 代入平面：A*(t*x) + B*(t*y) + C*(t*z) + D = 0
            # => t = -D / (A*x + B*y + C*z)
            denominator = np.dot(ray_dirs, normal)  # A*x + B*y + C*z
            valid_mask_ray = np.abs(denominator) > 1e-6
            t_vals = -D / (denominator + 1e-8)  # 避免除零
            t_vals = t_vals[valid_mask_ray]

            # 过滤有效交点
            ray_dirs = ray_dirs[valid_mask_ray]
            y_coords = y_coords[valid_mask_ray]
            x_coords = x_coords[valid_mask_ray]

            # 交点在相机坐标系下
            intersection_points = t_vals[:, None] * ray_dirs  # (M, 3)

            # 6. 提取这些交点的 Z 值（真实深度）
            true_depths_in_region = intersection_points[:, 2]  # (M,)

            # 同时过滤相对深度
            tag_region_depths_valid = relative_depth_map[mask > 0][valid_mask_ray]

            def cost_function(scale_factor):
                # 应用尺度因子调整相对深度值
                scaled_depths = tag_region_depths_valid * scale_factor
                
                # 计算误差平方和
                error = np.sum(np.abs(scaled_depths - true_depths_in_region))
                
                return error

            from scipy.optimize import minimize

            # 初始猜测尺度因子                    
            initial_guess = 1.0
            self.logger.info(f"开始优化AprilTag {i+1} 的尺度因子，初始值: {initial_guess}")
            result = minimize(cost_function, initial_guess)

            scale_factor = result.x[0]
            self.logger.info(f"AprilTag {i+1} 优化完成，尺度因子: {scale_factor:.6f}, 优化成功: {result.success}")

            scale_factors.append(scale_factor)

        # 7. 将相对深度图转换为绝对深度图
        optimal_scale_factor = np.mean(scale_factors)
        self.logger.info(f"所有AprilTag尺度因子: {[f'{sf:.6f}' for sf in scale_factors]}")
        self.logger.info(f"最优尺度因子: {optimal_scale_factor:.6f}")
        absolute_depth_map = relative_depth_map * optimal_scale_factor
        
        self.logger.info("深度恢复过程完成")
        return absolute_depth_map

    def estimate_depth(self, rgb_name):
        self.logger.info(f"开始深度估计，输入图像: {rgb_name}")
        depth_name = f'tmp/{os.path.basename(rgb_name).split(".")[0]}_depth.npy'
        self.logger.info(f"深度图将保存到: {depth_name}")
        
        # 确保tmp目录存在
        os.makedirs(os.path.dirname(depth_name), exist_ok=True)
         # our:8686,9393:depthanything,3909 depthcrafter
        process = subprocess.Popen([
            "python", 
            "GAVP/http_client/depth_estimator.py", 
            "--media_path", 
            rgb_name,
            "--download_path",
            depth_name,
            "--server_url=http://115.190.27.42:8686"
        ], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)

        # 实时记录输出
        self.logger.info("深度估计算法输出:")
        for line in process.stdout:
            self.logger.info(f"[深度估计] {line.strip()}")

        # 等待命令结束
        process.wait()
        if process.returncode != 0:
            error_msg = process.stderr.read()
            self.logger.error(f"深度估计命令执行失败，返回码: {process.returncode}")
            self.logger.error(f"错误信息: {error_msg}")
            return None
        
        self.logger.info("深度估计完成")
        return depth_name

    def run(self, rgb_name=None, depth_name=None, input_type="rgb", task_type="grasp_with_gripper"):
        
        self.logger.info("=" * 50)
        self.logger.info("开始执行GAVP管道流程")
        self.logger.info(f"输入类型: {input_type}, 任务类型: {task_type}")

        save_dir=None
        if rgb_name is None:
            save_dir = capture_one_frame(cfgs.save_dir)
            rgb_name= os.path.join(save_dir, 'color.png')
        else:
            save_dir = os.path.dirname(rgb_name)
        self.logger.info(f"加载RGB图像: {rgb_name}")
        rgb_image = cv2.imread(rgb_name)
        if rgb_image is None:
            self.logger.error(f"无法加载RGB图像: {rgb_name}")
            return False
        self.logger.info(f"RGB图像加载成功，尺寸: {rgb_image.shape}")
        
        # 2. 通过http访问算法，估计相对深度
        if depth_name is None:
            self.logger.info("开始相对深度估计")
            depth_name = self.estimate_depth(rgb_name)
            if depth_name is None:
                self.logger.error("深度估计失败")
                return False
        else:
            self.logger.info(f"使用提供的深度图: {depth_name}")
        
        self.logger.info(f"加载相对深度图: {depth_name}")
        disparity = np.load(depth_name)['depth'][0]
        relative_depth_map=disparity2depth(disparity)
        np.savez(os.path.join(save_dir, "relative_depth_map_14b.npz"), relative_depth_map=relative_depth_map)
        # relative_depth_map = disparity
        self.logger.info(f"相对深度图加载成功，形状: {relative_depth_map.shape}")

        # 3. 借助Apriltag恢复绝对深度
        self.logger.info("开始绝对深度恢复")
        try:
            depth_map = self.revover_depth(rgb_image, relative_depth_map)
        except Exception as e:
            self.logger.error(f"深度恢复失败: {str(e)}")
            return False
        
        
        # 3.1) 规整一下：确保和 RGB 同尺寸、float32、非法值置 0

        print(depth_map.shape)
        if depth_map.dtype != np.float32:
            depth_map = depth_map.astype(np.float32)
        np.savez(os.path.join(save_dir, "depth_map_14b.npz"), depth_map=depth_map)
        # 4. 调用本地Anygrasp，获得夹爪位姿TODO
        self.logger.info("开始夹爪位姿估计")
        try:
            R_cam, t_cam, width =run_anygrasp(save_dir,cfgs,rgb=rgb_image,depth=depth_map)
        except Exception as e:
            self.logger.error(f"夹爪位姿估计失败: {str(e)}")
            return False
        with open(os.path.join(save_dir, "grasp_pose.txt"), "w", encoding="utf-8") as f:
            f.write("R_cam:\n")
            f.write(np.array2string(R_cam, precision=6, suppress_small=True))
            f.write("\n\nt_cam:\n")
            f.write(np.array2string(t_cam.reshape(-1), precision=6, suppress_small=True))
            f.write(f"\n\nwidth: {width}\n")

        # 5. 发送夹爪位姿到下位机并收取是否成功的信息
        self.logger.info("发送夹爪位姿到下位机")
        run_pipline(R_cam, t_cam, width)


        self.logger.info(f"GAVP管道流程完成，结果: {'成功' if is_success else '失败'}")
        self.logger.info("=" * 50)
        return is_success

if __name__ == "__main__":
    # 设置主程序日志级别为DEBUG以获得更详细信息
    logging.getLogger().setLevel(logging.DEBUG)
    
    plat = AlgoValPlat(cam_params=[616.22601724,615.78839082,315.33494299,251.59150012])
    #rgb_name = "/home/wqz/all/桌子+透明/color.png"
    
    logger = logging.getLogger(__name__)
    logger.info("启动GAVP主程序")
    #logger.info(f"测试图像: {rgb_name}")
    
    is_success = plat.run()
    
    if is_success:
        logger.info("程序执行成功完成")
    else:
        logger.error("程序执行失败")
    
    logger.info(f"最终结果: {is_success}")