
import glob
import os
import pickle
from mmseg.apis import init_model, inference_model
import torch
from PIL import Image
import numpy as np
import torch

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

def get_image_paths_from_folder(folder_path):
    """
    获取并返回文件夹下所有图片的路径列表。

    参数:
    - folder_path: 文件夹路径

    返回值:
    - 图片路径列表
    """
    # 支持的图片扩展名
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 获取文件夹下所有文件的路径
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    
    # 过滤出图片文件
    image_paths = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(supported_extensions)]
    
    return image_paths

def get_image_paths_from_folder(folder_path):
    """
    获取并返回文件夹下所有图片的路径列表。

    参数:
    - folder_path: 文件夹路径

    返回值:
    - 图片路径列表
    """
    # 支持的图片扩展名
    supported_extensions = ('.jpg', '.jpeg', '.png', '.bmp')

    # 获取文件夹下所有文件的路径
    all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    
    # 过滤出图片文件
    image_paths = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(supported_extensions)]
    
    return image_paths

os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'

def infer_and_save(config_path, checkpoint_path, img_path, save_path, save_result=False, infer_one_by_one=False, save_as_label_type=False):
    """
    使用模型加载权重推理img_path所有的图像，并将推理结果保存为.png图片于save_path路径下。
    
    参数:
    - config_path: 模型配置文件路径
    - checkpoint_path: 模型权重文件路径
    - img_path: 图像路径，可以是单个图像路径，也可以是图像路径的列表
    - save_path: 将保存的.png格式图片保存的位置
    - save_result: 默认为False，将推理的得到的result对象保存为.pkl文件于save_path路径下
    - infer_one_by_one: 默认为False，是否逐个推理路径列表中的每个路径
    - save_as_label_type: 默认为False，是否直接将numpy数组保存为.png图像
    
    返回值:
    - 推理的Result对象列表或单个Result对象
    """
    
    # 检查img_path是否为列表，如果不是，则转换为列表
    if not isinstance(img_path, list):
        img_path = [img_path]
    
    # 初始化模型
    model = init_model(config_path, checkpoint_path)
    
    results = []
    failed_images = []  # 用于记录推理失败的图片路径和尺寸
    
    if infer_one_by_one:
        for img in img_path:
            try:
                # 读取图像以获取尺寸
                with Image.open(img) as image:
                    img_size = image.size
                
                # 推理单个图像
                result = inference_model(model, img)
                results.append(result)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Warning: CUDA out of memory for image {img} with size {img_size}, skipping this image.")
                    failed_images.append((img, img_size))
                    continue
                else:
                    raise e
    else:
        try:
            # 推理所有图像
            results = inference_model(model, img_path)
        except RuntimeError as e:
            if 'CUDA out of memory' in str(e):
                print("Warning: CUDA out of memory, unable to process the batch. Please try with infer_one_by_one=True.")
                for img in img_path:
                    with Image.open(img) as image:
                        img_size = image.size
                    failed_images.append((img, img_size))
                results = []
            else:
                raise e
    
    # 检查结果是否为单个对象，如果是，则转换为列表
    if not isinstance(results, list):
        results = [results]
    
    # 创建保存路径目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)
    
    for img, result in zip(img_path, results):
        if img in [i[0] for i in failed_images]:
            continue
        # 将推理结果的张量转换为NumPy数组
        cpu_tensor = result.pred_sem_seg.data.cpu()
        numpy_array = cpu_tensor.numpy()
        
        # 获取原图像名并去除扩展名
        img_name = os.path.basename(img).rsplit('.', 1)[0]
        save_img_path = os.path.join(save_path, f"{img_name}.png")
        
        if save_as_label_type:
            # 直接将NumPy数组保存为.png格式
            Image.fromarray(numpy_array[0, :, :].astype(np.uint8)).save(save_img_path)
        else:
            # 将NumPy数组转换为PIL图像
            image = Image.fromarray(numpy_array[0, :, :].astype(np.uint8) * 255)
            # 保存图像为.png格式
            image.save(save_img_path)
        
        print(f"Segmentation result saved as {save_img_path}")
    
    if save_result:
        # 将结果保存为.pkl文件
        results_pkl_path = os.path.join(save_path, 'results.pkl')
        with open(results_pkl_path, 'wb') as f:
            pickle.dump(results, f)
        print(f"Results object saved as {results_pkl_path}")
    
    if failed_images:
        # 记录推理失败的图片路径和尺寸到日志文件
        failed_log_path = os.path.join(save_path, 'failed_images.txt')
        with open(failed_log_path, 'w') as f:
            for img, img_size in failed_images:
                f.write(f"{img} with size {img_size}\n")
        print(f"Failed images logged in {failed_log_path}")
    
    return results
    
# # 示例调用
# config_path = r'/mnt/workspace/project/MMLAB/mmsegmentation/自动化脚本/辅助标注/hpc06061157_RFAinout_Dys_utp__mask2former_convnextv2-l.py'
# checkpoint_path = r'/mnt/workspace/project/MMLAB/mmsegmentation/自动化脚本/辅助标注/top_mIoU_97.7800_iter_3000.pth'
# # img_path = [r'/mnt/workspace/project/MMLAB/work_dirs/inference/170719STZ01_06.jpg', r"/mnt/workspace/project/MMLAB/work_dirs/inference/230628ETT02_08.jpg"]
# img_path = get_image_paths_from_folder(r"/mnt/workspace/project/MMLAB/mmsegmentation/my_mmseg_data/tissue推理的内容")

# save_path = r'/mnt/workspace/project/MMLAB/work_dirs/inference/rat_stomach_partial_data'
# infer_one_by_one = True
# results = infer_and_save(config_path, checkpoint_path, img_path, save_path, infer_one_by_one=infer_one_by_one)

def get_config_and_checkpoint(folder_path):
    """
    获取文件夹中的config和checkpoint文件路径。
    
    参数:
    - folder_path: 文件夹路径
    
    返回值:
    - config_path: 配置文件路径
    - checkpoint_path: 权重文件路径
    """
    config_files = glob.glob(os.path.join(folder_path, "*.py"))
    checkpoint_files = glob.glob(os.path.join(folder_path, "*.pth"))
    
    if len(config_files) != 1 or len(checkpoint_files) != 1:
        raise ValueError(f"文件夹 {folder_path} 中应包含一个 .py 配置文件和一个 .pth 权重文件。")
    
    return config_files[0], checkpoint_files[0]

def batch_infer_and_save(list_checkpoint_and_config_folder_path, img_path, save_path, save_result=False, infer_one_by_one=True, save_as_label_type=True):
    """
    对list_checkpoint_and_config_folder_path下的每个路径调用infer_and_save进行图像推理，并将结果保存到指定路径。
    
    参数:
    - list_checkpoint_and_config_folder_path: 包含多个checkpoint和config文件夹路径的列表
    - img_path: 图像路径或图像路径列表
    - save_path: 保存推理结果的根路径
    - save_result: 是否保存推理结果对象为.pkl文件，默认值为False
    - infer_one_by_one: 是否逐个推理路径列表中的每个路径，默认值为True
    - save_as_label_type: 是否直接将numpy数组保存为.png图像，默认值为True
    """
    for folder_path in list_checkpoint_and_config_folder_path:
        config_path, checkpoint_path = get_config_and_checkpoint(folder_path)
        
        # 确定保存路径
        specific_save_path = os.path.join(save_path, os.path.basename(folder_path))
        os.makedirs(specific_save_path, exist_ok=True)
        
        # 调用infer_and_save函数
        infer_and_save(config_path, checkpoint_path, img_path, specific_save_path, save_result, infer_one_by_one, save_as_label_type)

# 示例调用
list_checkpoint_and_config_folder_path = get_subdirectories(r"/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/ISIC最佳权重和配置文件")
img_path = get_image_paths_from_folder(r"/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/ISIC_2018_Task1/img_dir/test")
save_path = r'/home/sunhnayu/lln/project/MMLAB/work_dirs/infer/ISIC2018_test_set'

# list_checkpoint_and_config_folder_path = get_subdirectories(r"/mnt/workspace/project/MMLAB/mmsegmentation/my_mmseg_data/Ulcer最佳权重和配置文件")
# img_path = get_image_paths_from_folder(r"/mnt/workspace/project/MMLAB/mmsegmentation/my_mmseg_data/Rat_Ulcer/img_dir/test")
# save_path = r'/mnt/workspace/project/MMLAB/work_dirs/推理结果/Rat_Cerebral_Infarction_test_set_2'

batch_infer_and_save(list_checkpoint_and_config_folder_path, img_path, save_path)
