import os
import shutil
import math
import random

def split_dataset(dataset_folder_path, output_folder_path, copy_folder_name, base_number=2, selection_method="order"):
    """
    分割数据集并将其复制到新的文件夹结构中。

    参数:
    - dataset_folder_path: str, 数据集文件夹路径
    - output_folder_path: str, 目标内容的输出路径
    - copy_folder_name: str, 复制的dataset_folder_path子文件夹的子文件夹名，应该是”train”,”val”,”test”三者之一
    - base_number: int, 默认为2，复制文件时一次复制2^0,2^1,2^2,2^3...张图像
    - selection_method: str, 选择复制数据的方法，可以是 "max_interval", "order", "random"

    返回值:
    无
    """
    # 确保输出路径存在
    if not os.path.exists(output_folder_path):
        os.makedirs(output_folder_path)

    # 获取原始图像和标注文件的路径
    img_dir_path = os.path.join(dataset_folder_path, 'img_dir', copy_folder_name)
    ann_dir_path = os.path.join(dataset_folder_path, 'ann_dir', copy_folder_name)

    # 获取所有图像和标注文件
    img_files = sorted(os.listdir(img_dir_path))
    ann_files = sorted(os.listdir(ann_dir_path))

    # 检查图像和标注文件数量是否一致
    if len(img_files) != len(ann_files):
        raise ValueError("图像和标注文件数量不一致！")

    total_files = len(img_files)
    selected_indices = []

    i = 0
    while True:
        num_files_to_copy = base_number ** i
        if num_files_to_copy > total_files:
            num_files_to_copy = total_files

        if selection_method == "order":
            selected_indices = list(range(num_files_to_copy))
        elif selection_method == "max_interval":
            selected_indices = sorted(set(selected_indices + get_max_interval_indices(total_files, num_files_to_copy)))
        elif selection_method == "random":
            if num_files_to_copy > len(selected_indices):
                additional_indices = random.sample([idx for idx in range(total_files) if idx not in selected_indices], num_files_to_copy - len(selected_indices))
                selected_indices = sorted(selected_indices + additional_indices)
        else:
            raise ValueError("无效的选择方法")

        # 创建文件夹A的名称
        folder_name = f"{os.path.basename(dataset_folder_path)}_{copy_folder_name}_{num_files_to_copy}"
        folder_path = os.path.join(output_folder_path, folder_name)
        
        # 创建文件夹A以及子文件夹
        img_output_dir = os.path.join(folder_path, 'img_dir', copy_folder_name)
        ann_output_dir = os.path.join(folder_path, 'ann_dir', copy_folder_name)
        
        os.makedirs(img_output_dir, exist_ok=True)
        os.makedirs(ann_output_dir, exist_ok=True)

        # 复制文件
        for idx in selected_indices:
            shutil.copy(os.path.join(img_dir_path, img_files[idx]), img_output_dir)
            shutil.copy(os.path.join(ann_dir_path, ann_files[idx]), ann_output_dir)
        
        print(f"成功创建文件夹: {folder_name}, 复制了 {num_files_to_copy} 张图像和对应的标注文件")

        # 如果已经复制了所有文件，停止复制
        if num_files_to_copy == total_files:
            break

        i += 1

def get_max_interval_indices(total_files, num_files_to_copy):
    """
    获取最大间隔索引列表。

    参数:
    - total_files: int, 总文件数
    - num_files_to_copy: int, 需要复制的文件数

    返回值:
    - List[int], 索引列表
    """
    if num_files_to_copy >= total_files:
        return list(range(total_files))
    if num_files_to_copy == 1:
        return [0]

    indices = [0, total_files - 1]
    for _ in range(num_files_to_copy - 2):
        max_gap = 0
        insert_idx = 1
        for i in range(1, len(indices)):
            gap = indices[i] - indices[i-1]
            if gap > max_gap:
                max_gap = gap
                insert_idx = i
        indices.insert(insert_idx, (indices[insert_idx-1] + indices[insert_idx]) // 2)

    return indices


# 使用示例
# dataset_folder_path = 'path/to/dataset'
# output_folder_path = 'path/to/output'
# copy_folder_name = 'train'
# split_dataset(dataset_folder_path, output_folder_path, copy_folder_name, selection_method="max_interval")


# 使用示例
dataset_folder_path = r'E:\github\MMLAB\mmsegmentation\my_mmseg_data\ETIS_LaribPolypDB'
output_folder_path = r'E:\github\MMLAB\mmsegmentation\my_mmseg_data\ETIS_LaribPolypDB_sub'
copy_folder_name = 'train'
selection_method = "max_interval" # 可选 'order', "max_interval", "random"
split_dataset(dataset_folder_path, output_folder_path, copy_folder_name, selection_method=selection_method)

