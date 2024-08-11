import os
import time
import subprocess

def auto_test_single(test_py_path, config_path, checkpoint_path, 
                     dataset_root, dataset_type, 
                     test_batch_size, save_path, 
                     show=False, show_dir=None, wait_time=2):
    """
    简化的自动化测试脚本，用于在单卡机器上测试，并根据用户指定的参数构建测试命令并执行。

    参数:
    - test_py_path: 测试脚本路径。
    - config_path: 配置文件路径。
    - checkpoint_path: 模型权重文件路径。
    - dataset_root: 数据集根路径。
    - dataset_type: 数据集类型。
    - test_batch_size: 测试时的 batch size。
    - save_path: 保存路径。
    - show: 是否显示测试结果，默认值为 False。
    - show_dir: 保存可视化的分割掩膜图片的文件夹路径，默认值为 None。
    - wait_time: 多次可视化结果的时间间隔，默认值为 2。
    """
    # 创建日志文件夹
    log_dir_path = os.path.join(save_path, 'logs')
    os.makedirs(log_dir_path, exist_ok=True)
    
    # 获取当前时间作为日志文件名
    log_file_name = time.strftime("%Y%m%d_%H%M%S.txt", time.localtime())
    log_file_path = os.path.join(log_dir_path, log_file_name)

    with open(log_file_path, 'w') as log_file:
        # 记录当前配置文件的测试开始时间
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_file.write(f"开始测试配置文件: {config_path}, 开始时间: {start_time}\n")
        log_file.flush()

        # 构建基本测试命令
        command = f"python {test_py_path} {config_path} {checkpoint_path} --work-dir {save_path}"
        
        # 构建cfg-options字符串
        cfg_options = [
            f"test_dataloader.dataset.data_root={dataset_root}",
            f"test_dataloader.dataset.type={dataset_type}",
            f"test_dataloader.batch_size={test_batch_size}",
            f"work_dir={save_path}"
        ]
        if show:
            command += " --show"
        if show_dir is not None:
            command += f" --show-dir {show_dir}"
        command += f" --wait-time {wait_time}"
        
        # 添加cfg-options到命令中
        command += " --cfg-options " + " ".join(cfg_options)
        
        # 执行测试命令
        subprocess.call(command, shell=True)

        # 记录当前配置文件的测试结束时间
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_file.write(f"完成测试配置文件: {config_path}, 结束时间: {end_time}\n")
        log_file.flush()

# 示例用法
test_py_path = r"./MMLAB/mmsegmentation/tools/test.py"  # Path to MMLAB/mmsegmentation/tools/test.py
config_path = r"E:\github\CRMNet\MMLAB\mmsegmentation\my_mmseg_data\hpc06091027_RFAinout_Dys_RatUlcer__mask2former_convnextv2-l.py"  # Path to the configuration file
checkpoint_path = r"E:\HPC\最佳权重整理\Ulcer\hpc06091027_RFAinout_Dys_RatUlcer__mask2former_convnextv2-l_2\best_test_Ulcer_Dice_71.34_iter_3150.pth"  # Path to the checkpoint file
dataset_root = r"E:\github\MMLAB\mmsegmentation\my_mmseg_data\Rat_Ulcer"  # Root directory of the dataset
dataset_type = "MyDatasetUlcer"  # Dataset type, e.g., "MyDatasetUlcer" for "Rat_Ulcer", "MyDatasetCerebralInfarction" for "Rat_Cerebral_Infarction", "MyDatasetPolyp" for "CVC_ClinicDB", "CVC_ColonDB", "ETIS_LaribPolypDB" and "kvasir_seg", "MyDatasetISIC2018Task1" for "ISIC_2018_Task1"
test_batch_size = 1  # Test batch size
save_path = r".\work_dirs\hpc06091027"  # Path to save the test results
show = False  # Whether to show test results
show_dir = r".\work_dirs\timestamp"  # Directory to save visualized segmentation masks, only valid when show is False

auto_test_single(test_py_path, config_path, checkpoint_path, 
                 dataset_root=dataset_root, dataset_type=dataset_type, 
                 test_batch_size=test_batch_size, save_path=save_path, 
                 show=show, show_dir=show_dir)
