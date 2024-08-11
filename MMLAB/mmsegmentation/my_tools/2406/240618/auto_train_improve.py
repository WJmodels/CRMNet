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
    if not os.path.isdir(folder_path):
        raise ValueError("指定的文件夹路径不存在")

    matching_files = []
    for root, dirs, files in os.walk(folder_path):
        for file in files:
            if file.endswith(suffix):
                file_path = os.path.join(root, file)
                matching_files.append(file_path)

    return matching_files


def auto_train(list_config_path, log_dir_path, dist_train_script, cuda_visible_devices='0', port='29500', list_char_modify=None):
    """
    自动训练脚本

    参数:
    list_config_path (list): 配置文件路径列表
    log_dir_path (str): 日志文件夹路径
    dist_train_script (str): 分布式训练脚本路径
    cuda_visible_devices (str): CUDA设备编号字符串，默认值为 '0'
    port (str): 端口号，默认值为 '29500'
    list_char_modify (list): 配置修改字符串列表，默认值为 None (不使用)

    返回:
    无返回值
    """
    os.makedirs(log_dir_path, exist_ok=True)
    log_file_name = time.strftime("%Y%m%d_%H%M%S.txt", time.localtime())
    log_file_path = os.path.join(log_dir_path, log_file_name)

    num_devices = len(cuda_visible_devices.split(','))

    with open(log_file_path, 'w') as log_file:
        if list_char_modify is None:
            list_char_modify = [None] * len(list_config_path)
        elif len(list_config_path) == 1 and len(list_char_modify) > 1:
            list_config_path = list_config_path * len(list_char_modify)
        elif len(list_config_path) > 1 and len(list_char_modify) == 1:
            list_char_modify = list_char_modify * len(list_config_path)

        for config_path, char_modify in zip(list_config_path, list_char_modify):
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_file.write(f"开始训练配置文件: {config_path}, 开始时间: {start_time}\n")
            log_file.flush()

            command = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} PORT={port} bash {dist_train_script} {config_path} {num_devices}"
            if char_modify is not None:
                command += f" --cfg-options {char_modify}"
            
            try:
                subprocess.run(command, shell=True, check=True)
            except subprocess.CalledProcessError as e:
                log_file.write(f"训练配置文件失败: {config_path}, 错误信息: {str(e)}\n")

            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_file.write(f"完成训练配置文件: {config_path}, 结束时间: {end_time}\n")
            log_file.flush()

# 示例用法
# folder_path = r"/home/sunhnayu/lln/project/MMLAB/mmsegmentation/configs/classic_model"
# suffix = ".py"
# matching_files = get_files_with_suffix(folder_path, suffix)
# list_config_path = matching_files
list_config_path = [r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/configs/classic_model/hpc240613_deeplabv3plus_r50b-d8_4xb2-80k_cityscapes-512x1024.py',
                    r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/configs/classic_model/hpc240611_unet-s5-d16_deeplabv3_4xb4-40k_hrf-256x256.py']
print(list_config_path)

log_dir_path = r"/home/sunhnayu/lln/project/MMLAB/work_dirs/auto_train_log"
dist_train_script = r"/home/sunhnayu/lln/project/MMLAB/mmsegmentation/tools/dist_train.sh"
cuda_visible_devices = '0,1,2'
port = '29600'

# data_root = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/ETIS_LaribPolypDB'
# dataset_type = "MyDatasetPolyp"
data_root = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/ISIC_2018_Task1'
dataset_type = "MyDatasetISIC2018Task1"
max_iters = 20000
logger_interval = 10
val_interval = 100
train_batch_size = 6
val_batch_size = 1
test_batch_size = 1
char_modify = (
    f"default_hooks.logger.interval={logger_interval} "
    f"train_cfg.max_iters={max_iters} train_cfg.val_interval={val_interval} "
    f"train_dataloader.batch_size={train_batch_size} test_dataloader.batch_size={test_batch_size} val_dataloader.batch_size={val_batch_size} "
    f"train_dataloader.dataset.type='{dataset_type}' test_dataloader.dataset.type='{dataset_type}' val_dataloader.dataset.type='{dataset_type}' "
    f"train_dataloader.dataset.data_root='{data_root}' test_dataloader.dataset.data_root='{data_root}' val_dataloader.dataset.data_root='{data_root}'"
)
print(char_modify)
list_char_modify = [char_modify]

# 生成 train_data_root 的不同值
# i_values = [1, 2, 4, 8, 16, 32, 64, 128, 156]
# train_data_root_template = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/ETIS_LaribPolypDB_sub/ETIS_LaribPolypDB_train_{}'

# # 构建列表
# list_char_modify = [
#     (
#         f"train_cfg.max_iters={max_iters} train_cfg.val_interval={val_interval} "
#         f"train_dataloader.dataset.type='{dataset_type}' test_dataloader.dataset.type='{dataset_type}' val_dataloader.dataset.type='{dataset_type}' "
#         f"train_dataloader.dataset.data_root='{train_data_root_template.format(i)}' test_dataloader.dataset.data_root='{data_root}' val_dataloader.dataset.data_root='{data_root}' "
#         f"work_dir='../work_dirs/classic_model/ETIS_LaribPolypDB_sub/hpc06091027_RFAinout_Dys_RatUlcer__mask2former_convnextv2-l_{i}' "
#     )
#     for i in i_values
# ]

# # 输出生成的列表
# for item in list_char_modify:
#     print(item)

auto_train(list_config_path, log_dir_path, dist_train_script, cuda_visible_devices, port, list_char_modify)
