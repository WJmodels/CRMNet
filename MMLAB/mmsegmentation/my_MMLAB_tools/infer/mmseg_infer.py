from mmseg.apis import init_model

config_path = r'E:\编程\CODE\MMLAB自动化\训练权重和配置文件\ulcer_tissue_convnextv2_RFAinout_Dys_mask2former\hpc06061157_RFAinout_Dys_utp__mask2former_convnextv2-l.py'
checkpoint_path = r'E:\编程\CODE\MMLAB自动化\训练权重和配置文件\ulcer_tissue_convnextv2_RFAinout_Dys_mask2former\top_mIoU_97.7800_iter_3000.pth'

# 初始化不带权重的模型
model = init_model(config_path)

# 初始化模型并加载权重
model = init_model(config_path, checkpoint_path)

# 在 CPU 上的初始化模型并加载权重
model = init_model(config_path, checkpoint_path, 'cpu')