import os
import time
import subprocess

def auto_train_single(train_py_path, config_path, 
                      pretrain_checkpoint=None, dataset_root=None, dataset_type=None, 
                      train_batch_size=None, val_batch_size=None, test_batch_size=None, 
                      max_iters=None, val_interval=None, save_path=None):
    """
    简化的自动化训练脚本，用于在单卡机器上训练，并根据用户指定的参数构建训练命令并执行。

    参数:
    - train_py_path: 训练脚本路径。
    - config_path: 配置文件路径。
    - pretrain_checkpoint: 预训练模型路径，默认值为 None。
    - dataset_root: 数据集根路径，默认值为 None。
    - dataset_type: 数据集类型，默认值为 None。
    - train_batch_size: 训练时的batch size，默认值为 None。
    - val_batch_size: 验证时的batch size，默认值为 None。
    - test_batch_size: 测试时的batch size，默认值为 None。
    - max_iters: 最大迭代次数，默认值为 None。
    - val_interval: 验证间隔，默认值为 None。
    - save_path: 保存路径，默认值为 None。
    """
    # 创建日志文件夹
    log_dir_path = os.path.join(save_path, 'logs')
    os.makedirs(log_dir_path, exist_ok=True)
    
    # 获取当前时间作为日志文件名
    log_file_name = time.strftime("%Y%m%d_%H%M%S.txt", time.localtime())
    log_file_path = os.path.join(log_dir_path, log_file_name)

    with open(log_file_path, 'w') as log_file:
        # 记录当前配置文件的训练开始时间
        start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_file.write(f"开始训练配置文件: {config_path}, 开始时间: {start_time}\n")
        log_file.flush()

        # 构建基本训练命令
        command = f"python {train_py_path} {config_path} --work-dir {save_path}"
        
        # 构建cfg-options字符串
        cfg_options = []
        if pretrain_checkpoint is not None:
            cfg_options.append(f"load_from={pretrain_checkpoint}")
        if dataset_root is not None:
            cfg_options.extend([
                f"train_dataloader.dataset.data_root={dataset_root}",
                f"val_dataloader.dataset.data_root={dataset_root}",
                f"test_dataloader.dataset.data_root={dataset_root}"
            ])
        if dataset_type is not None:
            cfg_options.extend([
                f"train_dataloader.dataset.type={dataset_type}",
                f"val_dataloader.dataset.type={dataset_type}",
                f"test_dataloader.dataset.type={dataset_type}"
            ])
        if train_batch_size is not None:
            cfg_options.append(f"train_dataloader.batch_size={train_batch_size}")
        if val_batch_size is not None:
            cfg_options.append(f"val_dataloader.batch_size={val_batch_size}")
        if test_batch_size is not None:
            cfg_options.append(f"test_dataloader.batch_size={test_batch_size}")
        if max_iters is not None:
            cfg_options.append(f"train_cfg.max_iters={max_iters}")
        if val_interval is not None:
            cfg_options.append(f"train_cfg.val_interval={val_interval}")
        if save_path is not None:
            cfg_options.append(f"work_dir={save_path}")

        # 如果cfg_options不为空，则添加到命令中
        if cfg_options:
            command += " --cfg-options " + " ".join(cfg_options)
        
        # 执行训练命令
        subprocess.call(command, shell=True)

        # 记录当前配置文件的训练结束时间
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_file.write(f"完成训练配置文件: {config_path}, 结束时间: {end_time}\n")
        log_file.flush()

# 示例用法
train_py_path  = r"./MMLAB/mmsegmentation/tools/train.py"   # Path to MMLAB/mmsegmentation/tools/train.py
config_path = r"E:\github\CRMNet\MMLAB\mmsegmentation\my_mmseg_data\hpc06091027_RFAinout_Dys_RatUlcer__mask2former_convnextv2-l.py"    # Path to the configuration file
pretrain_checkpoint = r"E:\github\MMLAB\mmsegmentation\my_mmseg_pretrain_model\convnext-v2-large_fcmae-in21k-pre_3rdparty_in1k-384px_20230104-9139a1f3.pth"  # Path to the pre-trained model
dataset_root = r"E:\github\MMLAB\mmsegmentation\my_mmseg_data\Rat_Ulcer"   # Root directory of the dataset
dataset_type = "MyDatasetUlcer"        # Dataset type, e.g., "MyDatasetUlcer" for "Rat_Ulcer", 'MyDatasetCerebralInfarction' for "Rat_Cerebral_Infarction", 'MyDatasetPolyp' for "CVC_ClinicDB", "CVC_ColonDB", "ETIS_LaribPolypDB" and "kvasir_seg", 'MyDatasetISIC2018Task1' for "ISIC_2018_Task1"
train_batch_size = 4    # Training batch size
val_batch_size = 4     # Validation batch size
test_batch_size = 1  # Test batch size
max_iters = 20000    # Maximum number of training iterations
val_interval = 10    # Validation interval (number of iterations between validations)
save_path = r"E:\github\MMLAB\mmsegmentation\work_dirs"    # Path to save the training results



auto_train_single(train_py_path, config_path, 
                  pretrain_checkpoint=pretrain_checkpoint, dataset_root=dataset_root, 
                  dataset_type=dataset_type, train_batch_size=train_batch_size, 
                  val_batch_size=val_batch_size, test_batch_size=test_batch_size, 
                  max_iters=max_iters, val_interval=val_interval, save_path=save_path)
