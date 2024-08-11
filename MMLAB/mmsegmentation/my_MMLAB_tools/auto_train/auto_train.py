import os
import time
import subprocess

def auto_train(list_config_path, log_dir_path, dist_train_script, cuda_visible_devices='0', port='29500', char_modify=None):
    # 创建日志文件夹
    os.makedirs(log_dir_path, exist_ok=True)
    
    # 获取当前时间作为日志文件名
    log_file_name = time.strftime("%Y%m%d_%H%M%S.txt", time.localtime())
    log_file_path = os.path.join(log_dir_path, log_file_name)

    # 计算设备数量
    num_devices = len(cuda_visible_devices.split(','))

    with open(log_file_path, 'w') as log_file:
        for config_path in list_config_path:
            # 记录当前配置文件的训练开始时间
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_file.write(f"开始训练配置文件: {config_path}, 开始时间: {start_time}\n")
            log_file.flush()

            # 构建训练命令
            command = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} PORT={port} bash {dist_train_script} {config_path} {num_devices}"
            
            # 如果char_modify不为None,则添加--cfg-options
            if char_modify is not None:
                command += f" --cfg-options {char_modify}"
            
            # 执行训练命令
            subprocess.call(command, shell=True)

            # 记录当前配置文件的训练结束时间
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_file.write(f"完成训练配置文件: {config_path}, 结束时间: {end_time}\n")
            log_file.flush()


# 示例用法
list_config_path = [
    r"/mnt/workspace/project/MMLAB/mmsegmentation/configs/convnetv2/hpc05230848_mask2former_RFA_in_up_nooutconv_convnetv2-l.py",
    r"/mnt/workspace/project/MMLAB/mmsegmentation/configs/convnetv2/hpc05231608_mask2former_HAAMin_up_convnetv2-l.py",
]

log_dir_path = r"/mnt/workspace/project/MMLAB/work_dirs/自动化测试/日志"
dist_train_script = r"/mnt/workspace/project/MMLAB/mmsegmentation/tools/dist_train.sh"
cuda_visible_devices = '0'
port = '29600'
char_modify = "train_cfg.max_iters=10 train_cfg.val_interval=5"

auto_train(list_config_path, log_dir_path, dist_train_script, cuda_visible_devices, port, char_modify)