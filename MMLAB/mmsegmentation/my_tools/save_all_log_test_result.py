import re
import pandas as pd
import os
import numpy as np

def get_weight_file_path(log_file_path):
    """
    从日志文件中提取”Load checkpoint from ”后到换行前的内容
    
    参数:
    log_file_path (str): 日志文件路径
    
    返回值:
    weight_file_path (str): 从”Load checkpoint from ”到换行前的内容
    
    使用方法:
    weight_file_path = get_weight_file_path("path/to/log_file.log")
    """
    with open(log_file_path, 'r') as file:
        log_content = file.read()
    
    # 定义正则表达式匹配”Load checkpoint from ”后到换行前的内容
    match = re.search(r'Load checkpoint from (.+)', log_content)
    
    if match:
        weight_file_path = match.group(1).strip()
        return weight_file_path
    else:
        return None

def get_iter_from_weight_file_path(weight_file_path):
    """
    从权重文件路径中提取迭代次数
    
    参数:
    weight_file_path (str): 权重文件路径
    
    返回值:
    iter_num (int): 迭代次数
    
    使用方法:
    iter_num = get_iter_from_weight_file_path("/mnt/workspace/project/MMLAB/work_dirs/convnetv2/hpc05241555_topk_mask2former_RFAin_up_convnetv2-l.py/top_mIoU_7.7300_iter_20.pth")
    """
    # 定义正则表达式匹配迭代次数
    iter_match = re.search(r'iter_(\d+)\.pth', weight_file_path)
    
    if iter_match:
        iter_num = int(iter_match.group(1))
        return iter_num
    else:
        return None

def save_one_log_test_result(log_file_path, class_name, save_folder_path, save_as_csv=False, save_file=False):
    """
    从日志文件中提取指定类的测试评估指标值，并保存为Excel文件，增加weight_file_path列
    
    参数：
    log_file_path: 日志文件路径
    class_name: 类名，如 "colorectal_cancer"
    save_folder_path: 保存文件的文件夹路径
    save_as_csv: 是否保存为CSV文件，默认为False，保存为Excel文件
    save_file: 是否保存文件，默认为True，保存文件，False则只返回DataFrame对象
    
    返回值：
    生成的dataframe表格
    """
    # 获取权重文件路径
    weight_file_path = get_weight_file_path(log_file_path)
    
    # 获取评估指标字段
    list_evaluation_chars = get_list_evaluation_chars(log_file_path)
    
    # 获取类评估指标的具体值和对应的iter次数
    list_iter_list_class_evaluation_values = get_list_iter_list_class_evaluation_values(log_file_path, class_name, mode="test")
    
    if list_iter_list_class_evaluation_values is None or len(list_iter_list_class_evaluation_values) != 1:
        print("检测到多个test输出结果，请检查.log是否为test.py生成的文件")
        return None
    
    # 获取单条测试结果
    iter_val, values = list_iter_list_class_evaluation_values[0]
    
    # 生成DataFrame表格
    columns = ['Iter'] + list_evaluation_chars + ['weight_file_path']
    row = [iter_val] + values + [weight_file_path]
    df = pd.DataFrame([row], columns=columns)

    if save_file:
        if save_as_csv:
            # 获取日志文件名（不包括后缀.log）
            log_file_name = os.path.basename(log_file_path).rsplit('.', 1)[0]
            save_file_name = f"{log_file_name}_{class_name}_test.csv"
            save_file_path = os.path.join(save_folder_path, save_file_name)
            
            # 保存DataFrame为CSV文件
            df.to_csv(save_file_path, index=False)
            # 提示保存成功
            print(f"文件已成功保存到 {save_file_path}")
        else:
            # 获取日志文件名（不包括后缀.log）
            log_file_name = os.path.basename(log_file_path).rsplit('.', 1)[0]
            save_file_name = f"{log_file_name}_{class_name}_test.xlsx"
            save_file_path = os.path.join(save_folder_path, save_file_name)
            
            # 保存DataFrame为Excel文件
            df.to_excel(save_file_path, index=False)
            # 提示保存成功
            print(f"文件已成功保存到 {save_file_path}")
    
    return df

def get_list_evaluation_chars(log_file_path):
    with open(log_file_path, 'r') as f:
        log_content = f.read()

    # 定位"|       Class       |"所在行
    class_line = re.search(r'\|\s+Class\s+\|.*\n', log_content).group()

    # 用正则表达式匹配指标的字符串
    evaluation_chars = re.findall(r'\s+(\w+)\s+', class_line)
    evaluation_chars.remove('Class')

    return evaluation_chars

def get_list_iter_list_class_evaluation_values(log_file_path, target_class, mode="train"):
    """
    从.log文件中获取某类（如”colorectal_cancer”类）评估的指标的具体值，并与对应的iter次数组合成元组列表返回。
    
    参数:
    log_file_path (str): .log文件路径
    target_class (str): 具体的类名如”colorectal_cancer”
    mode (str): 匹配的是train.py生成的文件(mode="train")，还是test.py生成的文件(mode="test")，默认为“train”,
    
    返回值:
    list_iter_list_class_evaluation_values (list): 包含元组的列表，每个元组包含iter和评估指标值
    
    使用方法:
    result = get_list_iter_list_class_evaluation_values("path/to/log_file.log", "colorectal_cancer")
    """
    
    # 读取日志文件内容
    with open(log_file_path, 'r') as file:
        log_lines = file.readlines()
    
    # 用于存储结果的列表
    list_iter_list_class_evaluation_values = []
    current_iter = None
    
    # 定义正则表达式
    iter_pattern = re.compile(fr"Iter\({mode}\) \[ *(\d+)/\d+\]")
    class_pattern = re.compile(rf"{target_class} *\| *([\d\.]+|nan) *\| *([\d\.]+|nan) *\| *([\d\.]+|nan) *\| *([\d\.]+|nan) *\| *([\d\.]+|nan) *\| *([\d\.]+|nan)")

    # 遍历日志文件每一行
    for line in log_lines:
        # 查找iter
        iter_match = iter_pattern.search(line)
        if iter_match:
            current_iter = int(iter_match.group(1))
        
        # 查找目标类别的评估值
        class_match = class_pattern.search(line)
        if class_match and current_iter is not None:
            # 提取评估指标值
            values = [float(value) if value != 'nan' else np.nan for value in class_match.groups()]
            tuple_target = (current_iter, values)
            list_iter_list_class_evaluation_values.append(tuple_target)
    
    # 处理 mode 为 test 的情况
    if mode == "test":
        if len(list_iter_list_class_evaluation_values) != 1:
            print("检测到多个test输出结果，请检查.log是否为test.py生成的文件")
            return None
        weight_file_path = get_weight_file_path(log_file_path)
        if weight_file_path:
            iter_num = get_iter_from_weight_file_path(weight_file_path)
            if iter_num is not None:
                list_iter_list_class_evaluation_values[0] = (iter_num, list_iter_list_class_evaluation_values[0][1])
    
    return list_iter_list_class_evaluation_values

#################以上是工具函数，以下是主函数#################
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

def save_all_log_test_result(list_target_log_path, class_name, save_folder_path, save_as_csv=False):
    """
    遍历list_target_log_path中的每一个日志文件路径，调用save_one_log_test_result获取每个日志文件的DataFrame，
    并将所有DataFrame合并成一个总的DataFrame，然后保存为一个文件
    
    参数：
    list_target_log_path: 包含所有日志文件路径的列表
    class_name: 类名，如 "colorectal_cancer"
    save_folder_path: 保存文件的文件夹路径
    save_as_csv: 是否保存为CSV文件，默认为False，保存为Excel文件
    
    返回值：
    合并后的总的DataFrame表格
    """
    all_dfs = []
    
    for log_file_path in list_target_log_path:
        df = save_one_log_test_result(log_file_path, class_name, save_folder_path, save_as_csv=False)
        if df is not None:
            all_dfs.append(df)
    
    if all_dfs:
        combined_df = pd.concat(all_dfs, ignore_index=True)
        if save_as_csv:
            save_file_name = "combined_test_results.csv"
            save_file_path = os.path.join(save_folder_path, save_file_name)
            combined_df.to_csv(save_file_path, index=False)
            print(f"合并后的文件已成功保存到 {save_file_path}")
        else:
            save_file_name = "combined_test_results.xlsx"
            save_file_path = os.path.join(save_folder_path, save_file_name)
            combined_df.to_excel(save_file_path, index=False)
            print(f"合并后的文件已成功保存到 {save_file_path}")
        
        return combined_df
    else:
        print("未找到有效的测试结果")
        return None

if __name__ == '__main__':
    # 示例用法
    # 先获取目标日志文件路径
    list_target_log_path = find_target_logs('/mnt/workspace/project/MMLAB/work_dirs/convnetv2/hpc05241555_topk_mask2former_RFAin_up_convnetv2-l.py', 'Iter(test)')
    print(list_target_log_path)
    
    
    # 保存所有日志文件的测试结果
    class_name = 'colorectal_cancer'
    save_folder_path = '/mnt/workspace/project/MMLAB/work_dirs/convnetv2/hpc05241555_topk_mask2former_RFAin_up_convnetv2-l.py'
    # log_file_path = r'/mnt/workspace/project/MMLAB/work_dirs/convnetv2/hpc05241555_topk_mask2former_RFAin_up_convnetv2-l.py/20240604_182702/20240604_182702.log'
    # log_file_path = r'/mnt/workspace/project/MMLAB/work_dirs/convnetv2/hpc05241555_topk_mask2former_RFAin_up_convnetv2-l.py/20240604_182426/20240604_182426.log'
    # df = save_one_log_test_result(log_file_path, class_name, save_folder_path, save_as_csv=True)
    # print(df)
    
    combined_df = save_all_log_test_result(list_target_log_path, class_name, save_folder_path, save_as_csv=False)
    print(combined_df)
