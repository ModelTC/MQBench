import os
import random
import shutil
import glob

def copy_random_images(source_base_dir, dest_dir, num_per_class=5):
    """
    从一个类似 ImageNet 结构的源目录中，随机复制指定数量的图片到目标目录。

    Args:
        source_base_dir (str): 源图片库的基础目录 (例如: '.../tiny-imagenet-200/train')。
                               这个目录下应该包含多个类别的子文件夹。
        dest_dir (str): 要将图片复制到的目标目录。
        num_images (int): 要随机复制的图片数量。
    """
    print(f"任务开始：准备从 '{source_base_dir}' 随机复制图片。")

    # --- 第一步：准备目标目录 ---
    # 我们需要确保存放复制图片的目标文件夹是存在的。
    # os.makedirs(dest_dir, exist_ok=True) 是一个很方便的命令，
    # 它会创建文件夹，并且如果文件夹已经存在了，它也不会报错。
    try:
        os.makedirs(dest_dir, exist_ok=True)
        print(f"成功准备目标目录: '{dest_dir}'")
    except OSError as e:
        print(f"创建目录 '{dest_dir}' 时出错: {e}")
        return

    # --- 第二步：查找所有图片 ---
    # 我们需要遍历你给的源目录，找到所有 .JPEG 图片。
    # 你给的目录结构是 '.../train/<类别>/images/*.JPEG'
    # 我们可以用 glob 这个工具来轻松匹配所有符合这个结构的文件路径。
    class_dirs = [d for d in os.listdir(source_base_dir) if os.path.isdir(os.path.join(source_base_dir, d))]
    print(f"找到了 {len(class_dirs)} 个类别。")
    

    # --- 第三步：随机抽取图片 ---
    # 确定实际要复制的数量，防止想复制的比实际拥有的还多。
    total_copied_count = 0
    
    for i, class_name in enumerate(class_dirs):
        print(f"\r处理中: 类别 {i+1}/{len(class_dirs)} ({class_name})", end="")

        # 构建当前类别的图片文件夹路径
        source_class_images_dir = os.path.join(source_base_dir, class_name, 'images')

        # --- 第三步：查找当前类别的所有图片 ---
        if not os.path.isdir(source_class_images_dir):
            continue # 如果某个类别下没有 images 文件夹，就跳过

        image_search_pattern = os.path.join(source_class_images_dir, '*.JPEG')
        all_class_images = glob.glob(image_search_pattern)

        if not all_class_images:
            continue # 如果这个类别下没有图片，也跳过

        # --- 第四步：从当前类别中随机抽取图片 ---
        num_to_copy = min(num_per_class, len(all_class_images))
        selected_images = random.sample(all_class_images, num_to_copy)

        # --- 第五步：复制选中的图片，并保持结构 ---
        for src_path in selected_images:
            try:
                # 构建目标路径，这部分逻辑和上一版类似
                relative_path = os.path.relpath(src_path, source_base_dir)
                dest_path = os.path.join(dest_dir, relative_path)
                
                # 创建子目录
                dest_folder = os.path.dirname(dest_path)
                os.makedirs(dest_folder, exist_ok=True)
                
                # 复制文件
                shutil.copy2(src_path, dest_path)
                total_copied_count += 1
            except Exception as e:
                print(f"\n复制文件 {src_path} 时发生错误: {e}")

    print(f"\n\n任务完成！")
    print(f"总共处理了 {len(class_dirs)} 个类别。")
    print(f"成功复制了 {total_copied_count} 张图片到 '{dest_dir}'。")


# --- 程序主入口 ---
# 这是一个好习惯，把可以直接运行的代码放在这个 if 语句下面。
# 这样，当你把这个文件作为模块导入到其他文件中时，这部分代码就不会被执行。
if __name__ == '__main__':
    # --- 请在这里根据你的实际情况修改路径 ---
    
    # 源目录：就是包含所有分类文件夹 (如 n06596364) 的那个 'train' 目录
    SOURCE_DIRECTORY = '/mnt/share_disk/bruce_trie/outputs/tiny-imagenet-200/train'
    
    # 目标目录：你希望存放随机图片的那个文件夹
    DESTINATION_DIRECTORY = '/mnt/share_disk/bruce_trie/outputs/tiny-imagenet-200/train_mini'
    
    # 希望复制的图片数量
    NUMBER_OF_IMAGES_TO_COPY = 5

    # 调用我们上面定义的函数来执行复制操作
    copy_random_images(SOURCE_DIRECTORY, DESTINATION_DIRECTORY, NUMBER_OF_IMAGES_TO_COPY)