import os
import pickle
from mmseg.apis import init_model, inference_model
import torch
from PIL import Image
import numpy as np
import torch


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
    failed_images = []  # 用于记录推理失败的图片路径
    
    if infer_one_by_one:
        for img in img_path:
            try:
                # 推理单个图像
                result = inference_model(model, img)
                results.append(result)
            except RuntimeError as e:
                if 'CUDA out of memory' in str(e):
                    print(f"Warning: CUDA out of memory for image {img}, skipping this image.")
                    failed_images.append(img)
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
                failed_images = img_path
                results = []
            else:
                raise e
    
    # 检查结果是否为单个对象，如果是，则转换为列表
    if not isinstance(results, list):
        results = [results]
    
    # 创建保存路径目录（如果不存在）
    os.makedirs(save_path, exist_ok=True)
    
    for img, result in zip(img_path, results):
        if img in failed_images:
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
        # 记录推理失败的图片路径到日志文件
        failed_log_path = os.path.join(save_path, 'failed_images.txt')
        with open(failed_log_path, 'w') as f:
            for img in failed_images:
                f.write(f"{img}\n")
        print(f"Failed images logged in {failed_log_path}")
    
    return results

# 示例调用
img_path = get_image_paths_from_folder(r"/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/Rat_Cerebral_Infarction/img_dir/val")
# img_path = [r'/mnt/workspace/project/MMLAB/work_dirs/inference/170719STZ01_06.jpg', r"/mnt/workspace/project/MMLAB/work_dirs/inference/230628ETT02_08.jpg"]

# config_path = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/infer/Infarction_best_weight_and_config_file/hpc06091027_RFAinout_Dys_RatUlcer__mask2former_convnextv2-l/hpc06091027_RFAinout_Dys_RatUlcer__mask2former_convnextv2-l.py'
# checkpoint_path = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/infer/Infarction_best_weight_and_config_file/hpc06091027_RFAinout_Dys_RatUlcer__mask2former_convnextv2-l/best_test_Cerebral_Infarction_Dice_81.8_iter_5700.pth'
# save_path = r'/home/sunhnayu/lln/project/MMLAB/work_dirs/infer/Rat_Cerebral_Infarction_val_set/CRMNet'

# config_path = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/infer/Infarction_best_weight_and_config_file/hpc240611_unet-s5-d16_deeplabv3_4xb4-40k_hrf-256x256/hpc240611_unet-s5-d16_deeplabv3_4xb4-40k_hrf-256x256.py'
# checkpoint_path = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/infer/Infarction_best_weight_and_config_file/hpc240611_unet-s5-d16_deeplabv3_4xb4-40k_hrf-256x256/best_test_Cerebral_Infarction_Dice_79.37_iter_8600.pth'
# save_path = r'/home/sunhnayu/lln/project/MMLAB/work_dirs/infer/Rat_Cerebral_Infarction_val_set/Unet'

# config_path = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/infer/Infarction_best_weight_and_config_file/hpc240609_upernet_r101_4xb4-160k_ade20k-512x512/hpc240609_upernet_r101_4xb4-160k_ade20k-512x512.py'
# checkpoint_path = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/infer/Infarction_best_weight_and_config_file/hpc240609_upernet_r101_4xb4-160k_ade20k-512x512/best_test_Cerebral_Infarction_upernet_Dice_79.89_iter_4850.pth'
# save_path = r'/home/sunhnayu/lln/project/MMLAB/work_dirs/infer/Rat_Cerebral_Infarction_val_set/UperNet'

# config_path = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/infer/Infarction_best_weight_and_config_file/hpc240609_fcn_r101-d8_4xb4-160k_ade20k-512x512/hpc240609_fcn_r101-d8_4xb4-160k_ade20k-512x512.py'
# checkpoint_path = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/infer/Infarction_best_weight_and_config_file/hpc240609_fcn_r101-d8_4xb4-160k_ade20k-512x512/best_test_Cerebral_Infarction_Dice_71.09_iter_3350.pth'
# save_path = r'/home/sunhnayu/lln/project/MMLAB/work_dirs/infer/Rat_Cerebral_Infarction_val_set/FCN'

config_path = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/infer/Infarction_best_weight_and_config_file/hpc240609_deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512/hpc240609_deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512.py'
checkpoint_path = r'/home/sunhnayu/lln/project/MMLAB/mmsegmentation/my_mmseg_data/infer/Infarction_best_weight_and_config_file/hpc240609_deeplabv3plus_r101-d8_4xb4-160k_ade20k-512x512/best_test_Cerebral_Infarction_Dice_80.67_iter_6150.pth'
save_path = r'/home/sunhnayu/lln/project/MMLAB/work_dirs/infer/Rat_Cerebral_Infarction_val_set/DeepLabv3+'



infer_one_by_one = True
save_as_label_type = True
results = infer_and_save(config_path, checkpoint_path, img_path, save_path, infer_one_by_one=infer_one_by_one, save_as_label_type=save_as_label_type)
