import sys
import os
import shutil

def relative_to_absolute(relative_path):
    """
    将相对路径转换为绝对路径。
    参数:
        relative_path (str): 相对路径，可能包含Windows类型的分隔符。
    返回值:
        str: 绝对路径。
    """
    # 将Windows风格的分隔符替换为当前操作系统的分隔符
    if os.name == 'nt':  # Windows系统
        relative_path = relative_path.replace('/', '\\')
    else:  # 非Windows系统
        relative_path = relative_path.replace('\\', '/')
    # 处理相对路径中的用户目录符号 (~)
    expanded_path = os.path.expanduser(relative_path)
    # 规范化路径分隔符
    normalized_path = os.path.normpath(expanded_path)
    # 获取当前工作目录
    current_directory = os.getcwd()
    # 将相对路径转换为绝对路径
    if not os.path.isabs(normalized_path):
        absolute_path = os.path.join(current_directory, normalized_path)
    else:
        absolute_path = normalized_path
    absolute_path = os.path.abspath(absolute_path)
    print(f"Absolute path: {absolute_path}")
    return absolute_path
    
def find_checkpoint_hook():
    """
    查找当前Python环境中的checkpoint_hook.py文件路径。
    
    返回值:
        str: 如果找到文件，返回文件路径。
        None: 如果未找到文件，返回None。
    """
    # 获取当前Python解释器的路径
    python_executable_path = sys.executable

    # 获取Python环境根目录
    python_env_path = sys.prefix
    print(f"Python environment root: {python_env_path}")

    # 获取当前Python版本
    python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
    print(f"Current Python version: {python_version}")

    # 构建目标文件路径（方式一）
    target_file_path1 = os.path.join(python_env_path, 'lib', python_version, 'site-packages', 'mmengine', 'hooks', 'checkpoint_hook.py')
    print(f"Trying path 1: {target_file_path1}")

    # 构建目标文件路径（方式二）
    target_file_path2 = os.path.join(python_env_path, 'Lib', 'site-packages', 'mmengine', 'hooks', 'checkpoint_hook.py')
    print(f"Trying path 2: {target_file_path2}")

    # 检查目标文件是否存在
    if os.path.exists(target_file_path1):
        print(f"Found target file: {target_file_path1}")
        return target_file_path1
    elif os.path.exists(target_file_path2):
        print(f"Found target file: {target_file_path2}")
        return target_file_path2
    else:
        print("Target file not found")
        return None

# # 使用示例
# checkpoint_hook_path = find_checkpoint_hook()
# if checkpoint_hook_path:
#     print(f"目标文件存在：{checkpoint_hook_path}")
# else:
#     print("目标文件不存在")



def replace_file(source_path, target_path):
    """
    将目标路径target_path的文件替换为源路径source_path的文件。

    参数:
        source_path (str): 源文件路径。
        target_path (str): 目标文件路径。

    返回值:
        bool: 替换成功返回True，替换失败返回False。
    """
    try:
        if os.path.exists(target_path):
            shutil.copy(source_path, target_path)
            print(f"File successfully replaced: {target_path}")
            return True
        else:
            print(f"Target file does not exist: {target_path}")
            return False
    except Exception as e:
        print(f"Failed to replace file: {e}")
        return False

if __name__ == '__main__':
    # 使用示例
    source_file_path = find_checkpoint_hook()
    # print(source_file_path)
    target_file_path = r'E:\编程\CODE\MMLAB自动化\自动训练\checkpoint_hook.py'

    if replace_file(source_file_path, target_file_path):
        print("Replacement operation completed")
    else:
        print("Replacement operation failed")