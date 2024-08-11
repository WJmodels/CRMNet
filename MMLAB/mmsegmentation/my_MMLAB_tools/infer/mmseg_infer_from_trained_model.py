from mmseg.apis import init_model, inference_model

config_path = r"E:\编程\CODE\ulcer_index\model\hpc240609_unet-s5-d16_fcn_4xb4-160k_cityscapes-512x1024.py"
checkpoint_path = r"E:\编程\CODE\ulcer_index\model\best_test_Ulcer_Dice_78.63_iter_8950.pth"
img_path = r"C:\Users\15154\Desktop\溃疡大图像\230308ETA03_01.jpg"


model = init_model(config_path, checkpoint_path)
result = inference_model(model, img_path)

type(result.pred_sem_seg)
numpy_result = result.pred_sem_seg.data.to('cpu').numpy()[0]
print(type(numpy_result))
print(numpy_result.shape)
print(numpy_result)


import numpy as np
import cv2
import matplotlib.pyplot as plt

def generate_masks_from_array(numpy_array, carrier_index=2):
    """
    根据给定的numpy数组生成对应数量的mask。

    参数:
    numpy_array (np.ndarray): 输入的numpy数组。
    carrier_index (int): 指定的像素值，生成该像素值的mask时将所有非0区域视为carrier_index像素值的区域。默认值为2。

    返回:
    dict: 一个字典，字典中的键为numpy数组的像素值，值为生成的对应mask。
    """
    # 获取numpy数组中所有唯一的像素值
    unique_values = np.unique(numpy_array)
    
    # 创建一个字典来存储每个像素值对应的mask
    dict_numpy_mask = {}

    for value in unique_values:
        if value == 0:
            continue  # 跳过背景值（假设背景值为0）
        
        # 创建一个与输入数组相同形状的mask
        mask = np.zeros_like(numpy_array, dtype=np.uint8)
        
        # 将对应像素值的位置设置为255，其他位置保持为0
        mask[numpy_array == value] = 255
        
        # 如果当前处理的是carrier_index的mask，则将所有非0区域视为carrier_index像素值的区域
        if value == carrier_index:
            carrier_mask = np.zeros_like(numpy_array, dtype=np.uint8)
            carrier_mask[numpy_array != 0] = 255
            dict_numpy_mask[value] = carrier_mask
        else:
            dict_numpy_mask[value] = mask

    return dict_numpy_mask

def visualize_masks(dict_numpy_mask):
    """
    可视化生成的mask。

    参数:
    dict_numpy_mask (dict): 包含生成的mask的字典，键为像素值，值为对应的mask。
    """
    num_masks = len(dict_numpy_mask)
    fig, axes = plt.subplots(1, num_masks, figsize=(15, 5))
    
    if num_masks == 1:
        axes = [axes]  # 将axes转为列表以便于统一处理

    for i, (pixel_value, mask) in enumerate(dict_numpy_mask.items()):
        axes[i].imshow(mask, cmap='gray')
        axes[i].set_title(f'Pixel value: {pixel_value}')
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()



dict_numpy_mask = generate_masks_from_array(numpy_result, carrier_index=2)
visualize_masks(dict_numpy_mask)