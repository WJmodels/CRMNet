import os
import subprocess
from MMLAB.mmsegmentation.my_tools.add_save_best_and_save_top_k_function import replace_file, find_checkpoint_hook

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


def run_command(command):
    result = subprocess.run(command, shell=True, check=True)
    if result.returncode != 0:
        raise Exception(f"Command '{command}' failed with return code {result.returncode}")

def main():
    # 安装openmim和mmengine
    run_command("pip install -U openmim")
    run_command("mim install mmengine")
    
    # 安装指定版本的mmcv
    run_command('mim install "mmcv==2.0.1"')
    
    # 安装mmpretrain
    os.chdir("./MMLAB/mmpretrain")
    run_command("pip install -v -e .")
    
    # 安装mmdetection
    os.chdir("../mmdetection")
    run_command("pip install -v -e .")
    
    # 安装mmsegmentation
    os.chdir("../mmsegmentation")
    run_command("pip install -v -e .")
    
    # 安装ftfy
    run_command("pip install ftfy")

    # 安装regex
    run_command("pip install regex")

    # 替换文件
    replace_file(relative_to_absolute(r".\my_tools\changehook\checkpoint_hook.py"), find_checkpoint_hook())
    
    print("Environment setup completed successfully.")

if __name__ == "__main__":
    main()
