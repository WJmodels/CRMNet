import os
import shutil

def get_subdirectories(path):
    """
    获取指定路径下的一级子文件夹路径列表

    参数:
    path (str): 需要获取子文件夹的路径

    返回:
    list: 包含一级子文件夹路径的列表

    使用方法:
    subdirs = get_subdirectories('your_path_here')
    print(subdirs)
    """
    # 检查路径是否存在
    if not os.path.exists(path):
        raise ValueError(f"路径 '{path}' 不存在")

    # 获取路径下的所有子文件夹
    subdirectories = [os.path.join(path, name) for name in os.listdir(path) 
                      if os.path.isdir(os.path.join(path, name))]
    
    return subdirectories

import os
import shutil

def check_model_files(model_folder_path: str) -> (bool, str):
    """
    检查目标文件夹中的目标文件是否合理
    参数:
        model_folder_path: 模型文件夹路径
    返回:
        (bool, str): 检查结果和提示信息
    """
    # 获取文件列表
    files = os.listdir(model_folder_path)
    
    # 筛选目标.pth文件
    pth_files = [f for f in files if f.startswith('best_test_') and f.endswith('.pth')]
    # 筛选目标.py文件
    py_files = [f for f in files if f.endswith('.py')]
    
    # 检查文件数量
    if len(pth_files) != 1:
        return False, f"{model_folder_path}: 存在多个目标.pth文件或者没有找到目标.pth文件"
    if len(py_files) != 1:
        return False, f"{model_folder_path}: 存在多个目标.py文件或者没有找到目标.py文件"
    
    return True, (pth_files[0], py_files[0])

def copy_model_files(model_folder_path: str, save_folder_path: str):
    """
    将目标文件夹中的目标文件复制到指定路径下
    参数:
        model_folder_path: 模型文件夹路径
        save_folder_path: 目标保存路径
    """
    # 检查并获取目标文件
    is_valid, result = check_model_files(model_folder_path)
    
    # 如果检查不通过，抛出异常
    if not is_valid:
        raise ValueError(result)
    
    pth_file, py_file = result
    
    # 目标文件夹路径
    model_name = os.path.basename(model_folder_path)
    target_folder = os.path.join(save_folder_path, model_name)
    
    # 创建目标文件夹（如果不存在则创建）
    if not os.path.exists(target_folder):
        os.makedirs(target_folder)
    
    # 复制文件
    for file in [pth_file, py_file]:
        source_path = os.path.join(model_folder_path, file)
        destination_path = os.path.join(target_folder, file)
        shutil.copyfile(source_path, destination_path)
        print(f"文件 {file} 已成功复制到 {destination_path}")
    
    print(f"{model_folder_path} 所有文件复制完成")

def process_model_folders(list_model_folder_path: list, save_folder_path: str):
    """
    处理多个模型文件夹路径
    参数:
        list_model_folder_path: 模型文件夹路径列表
        save_folder_path: 目标保存路径
    """
    all_valid = True
    error_messages = []

    # 检查所有文件夹
    for model_folder_path in list_model_folder_path:
        valid, message = check_model_files(model_folder_path)
        if not valid:
            all_valid = False
            error_messages.append(message)
    
    # 如果有不符合的情况，输出提示信息
    if not all_valid:
        print("以下文件夹存在问题：")
        for message in error_messages:
            print(message)
        return
    
    # 如果所有检查都通过，执行复制操作
    for model_folder_path in list_model_folder_path:
        copy_model_files(model_folder_path, save_folder_path)




# 使用示例
list_model_folder_path = get_subdirectories(r"E:\HPC\classic_model\Ulcer")
# print(list_model_folder_path)
save_folder_path = r"E:\HPC\最佳权重整理\ulcer_2"

process_model_folders(list_model_folder_path, save_folder_path)
