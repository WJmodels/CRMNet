import os
import time
import subprocess

def auto_test(model_work_dir_path, config_path, log_dir_path, dist_test_script, cuda_visible_devices='0', port='29500'):
    """
    功能描述:
        该函数用于自动化测试mmsegmentation模型。它会遍历指定工作目录下的所有.pth权重文件,对每个权重文件进行测试,并将测试结果记录到日志文件中。
    
    参数解释:
        model_work_dir_path (str): 模型的工作路径,该路径下存放了多个.pth权重文件。
        config_path (str): 配置文件的路径,对应一个具体的配置文件。
        log_dir_path (str): 日志文件的保存路径,函数会在该路径下创建以函数运行时间命名的日志文件。
        dist_test_script (str): dist_test.sh文件的路径。
        cuda_visible_devices (str, optional): 可见的CUDA设备,默认为'0'。
        port (str, optional): 测试使用的端口,默认为'29500'。
    
    使用方法:
        1. 准备好模型的工作目录、配置文件、日志保存路径和dist_test.sh文件的路径。
        2. 根据需要设置可见的CUDA设备和端口号。
        3. 调用auto_test函数,传入相应的参数。
        4. 函数会自动遍历工作目录下的.pth文件,对每个文件进行测试,并将结果记录到日志文件中。
    """
    # 创建日志文件夹
    os.makedirs(log_dir_path, exist_ok=True)
    
    # 获取当前时间作为日志文件名
    log_file_name = time.strftime("%Y%m%d_%H%M%S.txt", time.localtime())
    log_file_path = os.path.join(log_dir_path, log_file_name)

    # 计算设备数量
    num_devices = len(cuda_visible_devices.split(','))

    # 获取工作目录下所有的.pth文件
    pth_files = [file for file in os.listdir(model_work_dir_path) if file.endswith('.pth')]

    with open(log_file_path, 'w') as log_file:
        log_file.write(f"配置文件路径: {config_path}\n")
        
        for pth_file in pth_files:
            # 构建权重文件的完整路径
            pth_file_path = os.path.join(model_work_dir_path, pth_file)
            
            # 记录当前权重文件的测试开始时间
            start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_file.write(f"开始测试权重文件: {pth_file}, 开始时间: {start_time}\n")
            log_file.flush()

            # 构建测试命令
            command = f"CUDA_VISIBLE_DEVICES={cuda_visible_devices} PORT={port} bash {dist_test_script} {config_path} {pth_file_path} {num_devices}"
            
            # 执行测试命令
            subprocess.call(command, shell=True)

            # 记录当前权重文件的测试结束时间
            end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
            log_file.write(f"完成测试权重文件: {pth_file}, 结束时间: {end_time}\n")
            log_file.flush()

# 示例用法
model_work_dir_path = r"/mnt/workspace/project/MMLAB/work_dirs/convnetv2/hpc05241555_topk_mask2former_RFAin_up_convnetv2-l.py"
config_path = r"/mnt/workspace/project/MMLAB/work_dirs/convnetv2/hpc05241555_topk_mask2former_RFAin_up_convnetv2-l.py/hpc05241555_topk_mask2former_RFAin_up_convnetv2-l.py"
log_dir_path = r"/mnt/workspace/project/MMLAB/work_dirs/自动化测试/日志/自动化test"
dist_test_script = r"/mnt/workspace/project/MMLAB/mmsegmentation/tools/dist_test.sh"
cuda_visible_devices = '0'
port = '29600'

auto_test(model_work_dir_path, config_path, log_dir_path, dist_test_script, cuda_visible_devices, port)