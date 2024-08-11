import os
import pickle
from mmseg.apis import init_model, inference_model
import torch
from PIL import Image
import numpy as np

import cv2
import matplotlib.pyplot as plt

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




def batch_visualize_number_label_segmentation_masks(png_path, out_ann_visual_path, color_palette=None, max_colors=60):
    """
    将单通道的标注图像转换为彩色图像以便于可视化,并将结果保存在指定路径。
    添加生成并保存颜色映射饼形图的功能。
    
    参数:
    - png_path: 包含标注格式PNG图像的路径。
    - out_ann_visual_path: 可视化图像的输出路径。
    - color_palette: 每个类别对应的颜色数组。每个颜色是一个(r, g, b)元组。
    - max_colors: 颜色映射的最大数量。默认为60。
    
    功能:
    - 读取标注图像,并根据颜色数组转换为彩色图像。
    - 如果未提供颜色数组,则自动生成颜色。
    - 保存彩色图像到指定的输出路径。
    - 在专门的子文件夹中生成并保存颜色映射饼形图。
    - 报告转换的状态。
    """
    # 确保输出路径存在
    if not os.path.exists(out_ann_visual_path):
        os.makedirs(out_ann_visual_path)
    
    # 创建专门存储饼图的子文件夹
    pie_chart_path = os.path.join(out_ann_visual_path, "pie_charts")
    if not os.path.exists(pie_chart_path):
        os.makedirs(pie_chart_path)

    # 初始化计数器
    success_count = 0
    failure_count = 0
    failure_files = []

    for filename in os.listdir(png_path):
        if filename.endswith(".png"):
            try:
                # 读取图像
                file_path = os.path.join(png_path, filename)
                mask = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
                if mask is None:
                    raise ValueError("Image could not be loaded.")

                # 确定颜色
                unique_classes = np.unique(mask)
                if color_palette is None or len(color_palette) < len(unique_classes):
                    # 自动生成颜色
                    color_maps = ['tab20', 'tab20b', 'tab20c']
                    color_palette = []
                    for cmap_name in color_maps:
                        cmap = plt.get_cmap(cmap_name)
                        colors = cmap(np.linspace(0, 1, cmap.N))[:, :3]
                        color_palette.extend(colors)
                    color_palette = (np.array(color_palette[:max_colors]) * 255).astype(np.uint8)

                # 创建彩色可视化图像
                visual_image = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8)
                for i, val in enumerate(unique_classes):
                    if val > 0:  # 跳过背景
                        visual_image[mask == val] = color_palette[i]

                # 保存彩色图像
                visual_path = os.path.join(out_ann_visual_path, f"visual_{filename}")
                cv2.imwrite(visual_path, visual_image)
                success_count += 1
                print(f"Visualization successful: {filename}")

                # 生成饼图
                class_counts = [np.sum(mask == val) for val in unique_classes]
                plt.figure(figsize=(8, 8))
                colors = [color_palette[i][::-1]/255.0 if val > 0 else (0, 0, 0) for i, val in enumerate(unique_classes)]
                plt.pie(class_counts, labels=unique_classes, colors=colors, autopct='%1.1f%%')
                plt.title('Pixel Value Distribution')
                plt.savefig(os.path.join(pie_chart_path, f"pie_chart_{filename}"))
                plt.close()

            except Exception as e:
                failure_count += 1
                failure_files.append(filename)
                print(f"Failed to visualize {filename}: {str(e)}")

    # 总结报告
    print(f"Total images visualized successfully: {success_count}")
    print(f"Total images failed to visualize: {failure_count}")
    if failure_count > 0:
        print("Failed visualizations:", failure_files)


def infer_and_save(config_path, checkpoint_path, img_path, save_path, save_result=False, infer_one_by_one=True, save_as_label_type=True):
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

    
    visual_path = os.path.join(save_path, 'visual')
    batch_visualize_number_label_segmentation_masks(save_path, visual_path)
    
    if failed_images:
        # 记录推理失败的图片路径和尺寸到日志文件
        failed_log_path = os.path.join(save_path, 'failed_images.txt')
        with open(failed_log_path, 'w') as f:
            for img, img_size in failed_images:
                f.write(f"{img} with size {img_size}\n")
        print(f"Failed images logged in {failed_log_path}")
    
    return results



config_path = r'E:\github\CRMNet\MMLAB\mmsegmentation\my_mmseg_data\hpc06091027_RFAinout_Dys_RatUlcer__mask2former_convnextv2-l.py'  # Path to the configuration file
checkpoint_path = r'E:\HPC\最佳权重整理\Ulcer_New\hpc06091027_RFAinout_Dys_RatUlcer__mask2former_convnextv2-l_2\best_test_Ulcer_Dice_56.67_iter_3250.pth'  # Path to the checkpoint file containing pre-trained model weights
img_folder_path = r"E:\github\MMLAB\mmsegmentation\my_mmseg_data\Rat_Ulcer\img_dir\test"  # Path to the folder containing test images
save_path = r'E:\github\CRMNet\work_dirs\2408082009'  # Path to save the inference results and visualizations



results = infer_and_save(config_path, checkpoint_path, get_image_paths_from_folder(img_folder_path), save_path)
