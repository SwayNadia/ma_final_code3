import os
import time
from viser_viewer import ViserViewer
import argparse

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--port", type=int, default=8080)
    args = parser.parse_args()
    
    # 创建viewer实例
    viewer = ViserViewer(args.data_dir)
    
    # 启动服务器
    viewer.start_server()
    
    # 监控数据目录并更新可视化
    last_step = -1
    while True:
        # 获取最新的数据文件
        data_files = sorted([
            f for f in os.listdir(args.data_dir)
            if f.startswith("viser_data_") and f.endswith(".npz")
        ])
        
        if data_files:
            latest_file = data_files[-1]
            current_step = int(latest_file.split("_")[-1].split(".")[0])
            
            if current_step > last_step:
                viewer.update_visualization(current_step)
                last_step = current_step
                
        time.sleep(1)  # 每秒检查一次

if __name__ == "__main__":
    main() 