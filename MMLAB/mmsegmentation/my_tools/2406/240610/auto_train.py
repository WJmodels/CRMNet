import os
import time
import subprocess


def get_files_with_suffix(folder_path, suffix):
    """
    获取指定文件夹中所有具有指定后缀的文件路径

    参数:
    folder_path (str): 文件夹路径
    suffix (str): 文件后缀（例如 '.txt'）

    返回:
    list: 包含所有匹配文件路径的字符串列表
    """
    # 检查文件夹路径是否存在
    if not os.path.isdir(folder_path):
        raise ValueError("指定的文件夹路径不存在")

    # 初始化结果列表
    matching_files = []

    # 遍历文件夹及其子文件夹中的所有文件
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                # 构建完整的文件路径
                file_path = os.path.join(root, file)
                matching_files.append(file_path)

    return matching_files



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
# 示例用法
folder_path = r"/home/sunhnayu/lln/project/MMLAB/mmsegmentation/configs/classic_model"
suffix = ".py"
matching_files = get_files_with_suffix(folder_path, suffix)
# print(matching_files)
# list_config_path = [
#     r"/home/sunhnayu/lln/project/MMLAB/mmsegmentation/configs/convnextv2/RFAinout_DySample_TTA/hpc06041658_RFAinout_Dys_ClinicDB_mask2former_convnextv2-l.py",
#     r"/home/sunhnayu/lln/project/MMLAB/mmsegmentation/configs/convnextv2/RFAinout_DySample_TTA/hpc06041658_RFAinout_Dys_ColonDB_mask2former_convnextv2-l.py",
#     r"/home/sunhnayu/lln/project/MMLAB/mmsegmentation/configs/convnextv2/RFAinout_DySample_TTA/hpc06041658_RFAinout_Dys_ETIS_mask2former_convnextv2-l.py",
#     r"/home/sunhnayu/lln/project/MMLAB/mmsegmentation/configs/convnextv2/RFAinout_DySample_TTA/hpc06041658_RFAinout_Dys_kvasir_unfreeze_mask2former_convnextv2-l.py",
# ]
list_config_path = matching_files
log_dir_path = r"/home/sunhnayu/lln/project/MMLAB/work_dirs/auto_train_log"
dist_train_script = r"//home/sunhnayu/lln/project/MMLAB/mmsegmentation/tools/dist_train.sh"
cuda_visible_devices = '0,1,2'
port = '29600'

# data_root = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/Rat_Ulcer'
# dataset_type = "MyDatasetUlcer"
data_root = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/Rat_Cerebral_Infarction'
dataset_type = "MyDatasetCerebralInfarction"
max_iters=10000
val_interval=50

# char_modify = f"train_cfg.max_iters=10 train_cfg.val_interval=5"+\
#               f"train_dataloader.dataset.type={dataset_type} test_dataloader.dataset.type={dataset_type} val_dataloader.dataset.type={dataset_type}"+\
#               f"train_dataloader.dataset.data_root={data_root} test_dataloader.dataset.data_root={data_root} val_dataloader.dataset.data_root={data_root}"+\
#               f"train_cfg.max_iters={max_iters} train_cfg.val_interval={val_interval}"

char_modify = (
    f"train_cfg.max_iters={max_iters} train_cfg.val_interval={val_interval} "
    f"train_dataloader.dataset.type='{dataset_type}' test_dataloader.dataset.type='{dataset_type}' val_dataloader.dataset.type='{dataset_type}' "
    f"train_dataloader.dataset.data_root='{data_root}' test_dataloader.dataset.data_root='{data_root}' val_dataloader.dataset.data_root='{data_root}'"
)
# print(char_modify)

auto_train(list_config_path, log_dir_path, dist_train_script, cuda_visible_devices, port, char_modify)