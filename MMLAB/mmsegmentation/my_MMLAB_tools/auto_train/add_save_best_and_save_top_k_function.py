import sys
import os
import shutil

# def find_checkpoint_hook():
#     """
#     查找当前Python环境中的checkpoint_hook.py文件路径。
    
#     返回值:
#         str: 如果找到文件，返回文件路径。
#         None: 如果未找到文件，返回None。
#     """
#     # 获取当前Python解释器的路径
#     python_executable_path = sys.executable

#     # 获取Python环境根目录
#     python_env_path = sys.prefix
#     print(f"Python environment root: {python_env_path}")

#     # 获取当前Python版本
#     python_version = f"python{sys.version_info.major}.{sys.version_info.minor}"
#     print(f"Current Python version: {python_version}")

#     # 构建目标文件路径（方式一）
#     target_file_path1 = os.path.join(python_env_path, 'lib', python_version, 'site-packages', 'mmengine', 'hooks', 'checkpoint_hook.py')
#     print(f"Trying path 1: {target_file_path1}")

#     # 构建目标文件路径（方式二）
#     target_file_path2 = os.path.join(python_env_path, 'Lib', 'site-packages', 'mmengine', 'hooks', 'checkpoint_hook.py')
#     print(f"Trying path 2: {target_file_path2}")

#     # 检查目标文件是否存在
#     if os.path.exists(target_file_path1):
#         print(f"Found target file: {target_file_path1}")
#         return target_file_path1
#     elif os.path.exists(target_file_path2):
#         print(f"Found target file: {target_file_path2}")
#         return target_file_path2
#     else:
#         print("Target file not found")
#         return None


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
        print("Target file not found, trying to find by walking through directories...")

    # 遍历当前环境的文件夹以查找目标文件
    for root, dirs, files in os.walk(python_env_path):
        for file in files:
            if file == 'checkpoint_hook.py':
                candidate_path = os.path.join(root, file)
                # 检查路径是否包含'mmengine/hooks'
                if 'mmengine' in candidate_path.replace('\\', '/').split('/') and 'hooks' in candidate_path.replace('\\', '/').split('/'):
                    print(f"Found target file by walking: {candidate_path}")
                    return candidate_path

    print("Target file not found in any method")
    return None



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