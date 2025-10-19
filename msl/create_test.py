import os
import shutil
import random
from sklearn.model_selection import train_test_split

def reorganize_cub_dataset(original_path, new_path, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15, seed=42):
    """
    重新组织CUB数据集，划分为train/val/test
    
    参数:
    original_path: 原始数据集路径（包含train和val文件夹）
    new_path: 新的数据集路径
    train_ratio: 训练集比例
    val_ratio: 验证集比例  
    test_ratio: 测试集比例
    seed: 随机种子
    """
    
    # 设置随机种子
    random.seed(seed)
    
    # 创建新的目录结构
    os.makedirs(new_path, exist_ok=True)
    for split in ['train', 'val', 'test']:
        os.makedirs(os.path.join(new_path, split), exist_ok=True)
    
    # 获取所有类别（假设train和val有相同的类别结构）
    train_path = os.path.join(original_path, 'train')
    val_path = os.path.join(original_path, 'val')
    
    # 检查原始目录是否存在
    if not os.path.exists(train_path) or not os.path.exists(val_path):
        print(f"错误: 在 {original_path} 中找不到train或val目录")
        return
    
    # 获取所有类别（过滤隐藏文件和文件夹）
    def get_valid_classes(path):
        classes = []
        for item in os.listdir(path):
            item_path = os.path.join(path, item)
            # 只保留目录且不是隐藏文件
            if os.path.isdir(item_path) and not item.startswith('.'):
                classes.append(item)
        return classes
    
    train_classes = get_valid_classes(train_path)
    val_classes = get_valid_classes(val_path)
    
    # 确保train和val的类别一致
    classes = sorted(list(set(train_classes) & set(val_classes)))
    print(f"找到 {len(classes)} 个有效类别")
    
    # 统计每个类别的图片数量
    total_images = 0
    class_image_counts = {}
    
    for class_name in classes:
        train_class_path = os.path.join(train_path, class_name)
        val_class_path = os.path.join(val_path, class_name)
        
        # 获取图片文件（过滤隐藏文件和非图片文件）
        def get_image_files(class_path):
            if not os.path.exists(class_path):
                return []
            files = []
            for file in os.listdir(class_path):
                file_path = os.path.join(class_path, file)
                # 只保留文件且不是隐藏文件，并且是常见图片格式
                if (os.path.isfile(file_path) and 
                    not file.startswith('.') and 
                    file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff'))):
                    files.append(file)
            return files
        
        train_images = get_image_files(train_class_path)
        val_images = get_image_files(val_class_path)
        
        all_images = train_images + val_images
        class_image_counts[class_name] = all_images
        total_images += len(all_images)
        
        print(f"类别 {class_name}: {len(train_images)} (train) + {len(val_images)} (val) = {len(all_images)} 张图片")
    
    print(f"\n总共 {total_images} 张图片")
    
    # 对每个类别进行划分
    for class_name, images in class_image_counts.items():
        if len(images) < 3:
            print(f"警告: 类别 {class_name} 只有 {len(images)} 张图片，跳过该类别")
            continue
        
        # 划分数据集
        train_imgs, temp_imgs = train_test_split(
            images, 
            test_size=(1 - train_ratio), 
            random_state=seed,
            shuffle=True
        )
        
        # 在剩余数据中划分val和test
        val_test_ratio = val_ratio / (val_ratio + test_ratio)
        val_imgs, test_imgs = train_test_split(
            temp_imgs, 
            test_size=(1 - val_test_ratio), 
            random_state=seed,
            shuffle=True
        )
        
        print(f"类别 {class_name}: train={len(train_imgs)}, val={len(val_imgs)}, test={len(test_imgs)}")
        
        # 复制文件到新目录
        copy_images(original_path, class_name, train_imgs, new_path, 'train')
        copy_images(original_path, class_name, val_imgs, new_path, 'val')
        copy_images(original_path, class_name, test_imgs, new_path, 'test')

def copy_images(original_path, class_name, image_list, new_path, split):
    """复制图片到新的目录结构"""
    
    # 在新目录中创建类别文件夹
    class_dir = os.path.join(new_path, split, class_name)
    os.makedirs(class_dir, exist_ok=True)
    
    # 查找图片在原始目录中的位置并复制
    copied_count = 0
    for img_name in image_list:
        # 先在train中找
        src_path = os.path.join(original_path, 'train', class_name, img_name)
        if not os.path.exists(src_path):
            # 在val中找
            src_path = os.path.join(original_path, 'val', class_name, img_name)
        
        if os.path.exists(src_path):
            dst_path = os.path.join(class_dir, img_name)
            shutil.copy2(src_path, dst_path)
            copied_count += 1
        else:
            print(f"警告: 找不到图片 {img_name}")
    
    return copied_count

def verify_dataset(new_path):
    """验证新数据集的完整性"""
    print("\n验证数据集...")
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(new_path, split)
        classes = [d for d in os.listdir(split_path) 
                  if os.path.isdir(os.path.join(split_path, d)) and not d.startswith('.')]
        total_images = 0
        
        print(f"\n{split} 集:")
        for class_name in classes:
            class_path = os.path.join(split_path, class_name)
            images = [f for f in os.listdir(class_path) 
                     if os.path.isfile(os.path.join(class_path, f)) and not f.startswith('.')]
            total_images += len(images)
            print(f"  {class_name}: {len(images)} 张图片")
        
        print(f"总计: {total_images} 张图片, {len(classes)} 个类别")

def cleanup_hidden_files(dataset_path):
    """清理数据集中的隐藏文件"""
    print("清理隐藏文件...")
    for root, dirs, files in os.walk(dataset_path):
        # 删除隐藏文件
        for file in files:
            if file.startswith('.'):
                file_path = os.path.join(root, file)
                os.remove(file_path)
                print(f"删除隐藏文件: {file_path}")
        
        # 删除隐藏目录
        for dir in dirs:
            if dir.startswith('.'):
                dir_path = os.path.join(root, dir)
                shutil.rmtree(dir_path)
                print(f"删除隐藏目录: {dir_path}")

if __name__ == "__main__":
    # 配置路径
    original_dataset_path = "../datasets/CUB-200"  # 请修改为您的实际路径
    new_dataset_path = "../datasets/CUB-200(+test)"  # 请修改为您想要保存的新路径
    
    # 划分比例
    TRAIN_RATIO = 0.7
    VAL_RATIO = 0.15
    TEST_RATIO = 0.15
    
    print("开始重新组织CUB数据集...")
    print(f"原始路径: {original_dataset_path}")
    print(f"新路径: {new_dataset_path}")
    print(f"划分比例: train={TRAIN_RATIO}, val={VAL_RATIO}, test={TEST_RATIO}")
    
    # 可选：先清理原始数据中的隐藏文件
    # cleanup_hidden_files(original_dataset_path)
    
    # 执行重新组织
    reorganize_cub_dataset(
        original_path=original_dataset_path,
        new_path=new_dataset_path,
        train_ratio=TRAIN_RATIO,
        val_ratio=VAL_RATIO,
        test_ratio=TEST_RATIO,
        seed=42
    )
    
    # 验证结果
    verify_dataset(new_dataset_path)
    
    print("数据集重新组织完成！")