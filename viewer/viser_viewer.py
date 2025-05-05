import viser
import numpy as np
import os
import time
from pathlib import Path

class ViserViewer:
    def __init__(self, data_dir):
        self.data_dir = data_dir
        self.server = viser.ViserServer()
        self.point_cloud = None
        self.camera_poses = []
        
    def start_server(self):
        """启动viser服务器"""
        self.server.start()
        
    def load_data(self, step):
        """加载指定步数的数据"""
        data_path = os.path.join(self.data_dir, f"viser_data_{step:06d}.npz")
        if os.path.exists(data_path):
            data = np.load(data_path)
            return data
        return None
        
    def update_visualization(self, step):
        """更新可视化"""
        data = self.load_data(step)
        if data is None:
            return
            
        # 更新点云
        if self.point_cloud is not None:
            self.server.remove_point_cloud(self.point_cloud)
            
        self.point_cloud = self.server.add_point_cloud(
            points=data["points"],
            colors=data["colors"]
        )
        
        # 更新相机位姿
        for pose in self.camera_poses:
            self.server.remove_camera(pose)
        self.camera_poses = []
        
        camera_pose = data["camera_pose"]
        self.camera_poses.append(
            self.server.add_camera(
                position=camera_pose[:3],
                rotation=camera_pose[3:],
                fov=60.0
            )
        ) 