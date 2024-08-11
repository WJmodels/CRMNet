import re
import pandas as pd
import os

def get_list_evaluation_chars(log_file_path):
    with open(log_file_path, 'r') as f:
        log_content = f.read()

    # 定位"|       Class       |"所在行
    class_line = re.search(r'\|\s+Class\s+\|.*\n', log_content).group()

    # 用正则表达式匹配指标的字符串
    evaluation_chars = re.findall(r'\s+(\w+)\s+', class_line)
    evaluation_chars.remove('Class')

    return evaluation_chars


# def get_list_iter_list_class_evaluation_values(log_file_path, target_class, mode="train"):
#     """
#     从.log文件中获取某类（如”colorectal_cancer”类）评估的指标的具体值，并与对应的iter次数组合成元组列表返回。
    
#     参数:
#     log_file_path (str): .log文件路径
#     target_class (str): 具体的类名如”colorectal_cancer”
#     mode (str): 匹配的是train.py生成的文件(mode="train")，还是test.py生成的文件(mode="test")，默认为“train”,
    
#     返回值:
#     list_iter_list_class_evaluation_values (list): 包含元组的列表，每个元组包含iter和评估指标值
    
#     使用方法:
#     result = get_list_iter_list_class_evaluation_values("path/to/log_file.log", "colorectal_cancer")
#     """
    
#     # 读取日志文件内容
#     with open(log_file_path, 'r') as file:
#         log_lines = file.readlines()
    
#     # 用于存储结果的列表
#     list_iter_list_class_evaluation_values = []
#     current_iter = None
    
#     # 定义正则表达式
#     # iter_pattern = re.compile(r"Iter\(train\) \[ *(\d+)/\d+\]")
#     iter_pattern = re.compile(fr"Iter\({mode}\) \[ *(\d+)/\d+\]")
#     class_pattern = re.compile(rf"{target_class} *\| *([\d\.]+) *\| *([\d\.]+) *\| *([\d\.]+) *\| *([\d\.]+) *\| *([\d\.]+) *\| *([\d\.]+)")

#     # 遍历日志文件每一行
#     for line in log_lines:
#         # 查找iter
#         iter_match = iter_pattern.search(line)
#         if iter_match:
#             current_iter = int(iter_match.group(1))
        
#         # 查找目标类别的评估值
#         class_match = class_pattern.search(line)
#         if class_match and current_iter is not None:
#             # 提取评估指标值
#             iou, acc, dice, fscore, precision, recall = map(float, class_match.groups())
#             list_class_evaluation_values = [iou, acc, dice, fscore, precision, recall]
#             tuple_target = (current_iter, list_class_evaluation_values)
#             list_iter_list_class_evaluation_values.append(tuple_target)
    
#     return list_iter_list_class_evaluation_values
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
    class_pattern = re.compile(rf"{target_class} *\| *([\d\.]+) *\| *([\d\.]+) *\| *([\d\.]+) *\| *([\d\.]+) *\| *([\d\.]+) *\| *([\d\.]+)")

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
            iou, acc, dice, fscore, precision, recall = map(float, class_match.groups())
            list_class_evaluation_values = [iou, acc, dice, fscore, precision, recall]
            tuple_target = (current_iter, list_class_evaluation_values)
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

def save_one_class_evaluation_result(log_file_path, class_name, save_folder_path, save_as_csv=False,mode="train"):
    """
    从日志文件中提取指定类的评估指标值，并保存为Excel文件
    参数：
    log_file_path: 日志文件路径
    class_name: 类名，如 "colorectal_cancer"
    save_folder_path: 保存文件的文件夹路径
    
    返回值：
    生成的dataframe表格
    """
    # 获取评估指标字段
    list_evaluation_chars = get_list_evaluation_chars(log_file_path)
    
    # 获取类评估指标的具体值和对应的iter次数
    list_iter_list_class_evaluation_values = get_list_iter_list_class_evaluation_values(log_file_path, class_name,mode=mode)
    
    # 生成DataFrame表格
    columns = ['Iter'] + list_evaluation_chars
    data = []
    for iter_val, values in list_iter_list_class_evaluation_values:
        row = [iter_val] + values
        data.append(row)
    
    df = pd.DataFrame(data, columns=columns)

    if save_as_csv:
        # 获取日志文件名（不包括后缀.log）
        log_file_name = os.path.basename(log_file_path).rsplit('.', 1)[0]
        save_file_name = f"{log_file_name}_{class_name}.xlsx"
        save_file_path = os.path.join(save_folder_path, save_file_name)
        
        # 保存DataFrame为Excel文件
        df.to_excel(save_file_path, index=False)
        # 提示保存成功
        print(f"文件已成功保存到 {save_file_path}")
    
    return df


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


import pandas as pd

def save_one_log_test_result(log_file_path, class_name, save_folder_path, save_as_csv=False):
    """
    从日志文件中提取指定类的测试评估指标值，并保存为Excel文件，增加weight_file_path列
    
    参数：
    log_file_path: 日志文件路径
    class_name: 类名，如 "colorectal_cancer"
    save_folder_path: 保存文件的文件夹路径
    save_as_csv: 是否保存为CSV文件，默认为False，保存为Excel文件
    
    返回值：
    生成的dataframe表格
    """
    # 获取权重文件路径
    weight_file_path = get_weight_file_path(log_file_path)
    
    # 获取评估指标字段
    list_evaluation_chars = get_list_evaluation_chars(log_file_path)
    
    # 获取类评估指标的具体值和对应的iter次数
    list_iter_list_class_evaluation_values = get_list_iter_list_class_evaluation_values(log_file_path, class_name, mode="test")
    
    if len(list_iter_list_class_evaluation_values) != 1:
        print("检测到多个test输出结果，请检查.log是否为test.py生成的文件")
        return None
    
    # 获取单条测试结果
    iter_val, values = list_iter_list_class_evaluation_values[0]
    
    # 生成DataFrame表格
    columns = ['Iter'] + list_evaluation_chars + ['weight_file_path']
    row = [iter_val] + values + [weight_file_path]
    df = pd.DataFrame([row], columns=columns)

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

# 示例用法
# log_file_path = 'path/to/log_file.log'
# class_name = 'colorectal_cancer'
# save_folder_path = 'path/to/save_folder'
# df = save_one_log_test_result(log_file_path, class_name, save_folder_path, save_as_csv=False)
# print(df)


if __name__ == '__main__':
    save_folder_path = r'C:\Users\15154\Desktop\咕泡'
    # log_file_path = r"E:\编程\CODE\MMLAB自动化\log文件\20240604_172927.log"
    log_file_path = r"E:\编程\CODE\MMLAB自动化\log文件\test20240604_182733.log"

    # 获取评估指标列表
    list_evaluation_chars = get_list_evaluation_chars(log_file_path)
    print(list_evaluation_chars)

    # # 获取某个类在不同迭代次数下的评估指标值
    # class_name = 'Polyp'
    class_name = 'colorectal_cancer'
    
    # mode = "train"
    mode = "test"
    list_iter_list_class_evaluation_values = get_list_iter_list_class_evaluation_values(log_file_path, class_name,mode=mode)
    
    print(list_iter_list_class_evaluation_values)
    print(len(list_iter_list_class_evaluation_values))

    # save_one_class_evaluation_result 使用示例
    # df = save_one_class_evaluation_result(log_file_path, class_name, save_folder_path, save_as_csv=True)
    # df.head()

    # # save_one_log_test_result 使用示例
    # df = save_one_log_test_result(log_file_path, class_name, save_folder_path, save_as_csv=False)
    # print(df)

    # # 使用示例
    # log_file_path = r'E:\编程\CODE\MMLAB自动化\log文件\test20240604_182733.log'
    # weight_file_path = get_weight_file_path(log_file_path)
    # print(weight_file_path)

    # iter_num = get_iter_from_weight_file_path(weight_file_path)
    # print(iter_num)