import pandas as pd
import os
from datetime import datetime
import os

# def get_file_paths_from_folder(folder_path, file_extensions):
#     """
#     获取并返回文件夹下所有指定后缀文件的路径列表。

#     参数:
#     - folder_path: 文件夹路径
#     - file_extensions: 文件后缀的元组, 如 ('.jpg', '.png')

#     返回值:
#     - 指定后缀文件的路径列表
#     """
#     # 将文件后缀统一为小写
#     file_extensions = tuple(ext.lower() for ext in file_extensions)
    
#     # 获取文件夹下所有文件的路径
#     all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
    
#     # 过滤出指定后缀的文件
#     file_paths = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(file_extensions)]
    
#     return file_paths

import os

def get_file_paths_from_folder(folder_path, file_extensions, recursive=True):
    """
    获取并返回文件夹下所有指定后缀文件的路径列表。

    参数:
    - folder_path: 文件夹路径
    - file_extensions: 文件后缀的元组, 如 ('.jpg', '.png')
    - recursive: 是否递归搜索子文件夹，默认值为 False

    返回值:
    - 指定后缀文件的路径列表
    """
    # 将文件后缀统一为小写
    file_extensions = tuple(ext.lower() for ext in file_extensions)
    
    file_paths = []
    
    if recursive:
        # 递归遍历文件夹及其子文件夹
        for root, dirs, files in os.walk(folder_path):
            for file in files:
                file_path = os.path.join(root, file)
                if file.lower().endswith(file_extensions):
                    file_paths.append(file_path)
    else:
        # 获取文件夹下所有文件的路径
        all_files = [os.path.join(folder_path, f) for f in os.listdir(folder_path)]
        # 过滤出指定后缀的文件
        file_paths = [f for f in all_files if os.path.isfile(f) and f.lower().endswith(file_extensions)]
    
    return file_paths

def merge_and_summarize_excel_files(list_xlsx_file_path, suffix=".xlsx", max_item="Dice", output_folder_path="."):
    """
    汇总每个表格的内容，将每个sheet中”max_item”值最大的条目汇总到一个表格中，并最左侧添加上一列”sheet_name”表示汇总的这个条目出自哪个sheet。

    参数:
    list_xlsx_file_path (list): 一个包含.xlsx文件路径的列表。
    suffix (str): 文件后缀，默认为".xlsx"。
    max_item (str): 选出”max_item”最大的条目汇总到一个表格中, 默认”Dice”。
    output_folder_path (str): 保存输出.xlsx文件的文件夹路径, 输出文件命名为merge_{时间}。

    返回值:
    str: 输出文件的路径。
    """
    sheets_content = []
    max_item_summary = pd.DataFrame()
    
    for file_path in list_xlsx_file_path:
        file_name = os.path.basename(file_path).replace(suffix, "")[:31]
        
        xls = pd.ExcelFile(file_path)
        for sheet in xls.sheet_names:
            sheet_df = pd.read_excel(file_path, sheet_name=sheet)
            sheets_content.append((file_name, sheet_df))
            
            max_item_row = sheet_df.loc[sheet_df[max_item].idxmax()]
            max_item_row["sheet_name"] = file_name
            max_item_row = pd.DataFrame(max_item_row).T
            max_item_summary = pd.concat([max_item_summary, max_item_row], ignore_index=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file_path = os.path.join(output_folder_path, f"merge_{timestamp}.xlsx")
    
    with pd.ExcelWriter(output_file_path) as writer:
        for sheet_name, content in sheets_content:
            content.to_excel(writer, sheet_name=sheet_name, index=False)
        
        max_item_summary.to_excel(writer, sheet_name=f"Max_{max_item}_Summary", index=False)
    
    return output_file_path

# 示例用法
# list_xlsx_file_path = [
#     "/mnt/data/test_ClinicDB_Polyp_20240605_224152.xlsx",
#     "/mnt/data/test_ColonDB_Polyp_20240605_225141.xlsx",
#     "/mnt/data/test_ETIS_Polyp_20240605_223629.xlsx",
#     "/mnt/data/test_kvasir_Polyp_20240605_224714.xlsx"
# ]

folder_path = r"E:\HPC\最佳权重整理\Ulcer_New\excel"
file_extensions = '.xlsx'
list_xlsx_file_path = get_file_paths_from_folder(folder_path, file_extensions)
print(list_xlsx_file_path)

output_folder_path = r"E:\HPC\最佳权重整理\Ulcer_New\excel\merge"
merged_file_path = merge_and_summarize_excel_files(list_xlsx_file_path, output_folder_path=output_folder_path)
print(f"合并后的文件路径: {merged_file_path}")
