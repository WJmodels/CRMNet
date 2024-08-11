import os

def find_target_logs(model_work_dir_path, search_char='Iter(test)'):
    """
    函数功能描述：
    遍历指定路径下的所有.log文件，找出内容中包含目标字符串的.log文件，并返回这些文件的路径列表。

    函数参数解释：
    - model_work_dir_path: 模型的工作路径，字符串类型
    - search_char: 目标字符串，默认值为'Iter(test)'，字符串类型

    函数使用方法说明：
    - 调用函数时，传入模型的工作路径(model_work_dir_path)和目标字符串(search_char，可选)
    - 函数会返回一个包含目标.log文件路径的列表(list_target_log_path)

    示例：
    list_target_log_path = find_target_logs('/path/to/model/work/dir', 'Iter(test)')
    """
    list_target_log_path = []

    # 遍历model_work_dir_path路径下的所有文件和子文件夹
    for root, dirs, files in os.walk(model_work_dir_path):
        # 遍历当前文件夹下的所有文件
        for file in files:
            # 检查文件是否以.log结尾
            if file.endswith('.log'):
                log_file_path = os.path.join(root, file)
                # 读取.log文件内容
                with open(log_file_path, 'r') as f:
                    log_content = f.read()
                # 检查文件内容是否包含目标字符串
                if search_char in log_content:
                    list_target_log_path.append(log_file_path)

    return list_target_log_path

if __name__ == '__main__':
    # 示例用法
    list_target_log_path = find_target_logs('/mnt/workspace/project/MMLAB/work_dirs/convnetv2/hpc05241555_topk_mask2former_RFAin_up_convnetv2-l.py', 'Iter(test)')
    print(list_target_log_path)