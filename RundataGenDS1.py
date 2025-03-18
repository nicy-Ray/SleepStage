import numpy as np
# from DataGenerator import kFoldGenerator, DominGenerator
from isrucutitls.DataGenerator import kFoldGenerator, DominGenerator

# 加载数据
data = np.load("ISRUC_S3_all.npy")
fold_data = data["Fold_data"]   # 形状假设: (k_folds, num_samples, features)
fold_label = data["Fold_label"] # 形状假设: (k_folds, num_samples, num_classes)
fold_len = data["Fold_len"]     # 每个 fold 的原始序列长度 (可能用于窗口生成)
k = fold_data.shape[0]          # 总 fold 数（如 10 折交叉验证）

# 打印数据结构确认
print("Fold Data Shape:", fold_data.shape)
print("Fold Label Shape:", fold_label.shape)
print("Fold Lengths:", fold_len)

window_size = 5  # 窗口大小（2*T+1=5，T=2）
x_folds, y_folds = [], []

for i in range(k):
    # 提取第 i 个 fold 的原始时序数据
    x_subject = fold_data[i]   # 形状: (原始长度, features)
    y_subject = fold_label[i]  # 形状: (原始长度, num_classes)
    L = x_subject.shape[0]     # 原始时间点数量
    
    # 生成滑动窗口（重叠窗口）
    num_windows = L - window_size + 1
    x_windows = np.array([x_subject[t:t+window_size] for t in range(num_windows)])
    y_windows = y_subject[window_size//2 : window_size//2 + num_windows]  # 取窗口中心标签
    
    x_folds.append(x_windows)
    y_folds.append(y_windows)

# 示例输出
print(f"Fold 0 窗口数据形状: {x_folds[0].shape}")  # 应为 (num_windows, window_size, features)
print(f"Fold 0 窗口标签形状: {y_folds[0].shape}")  # 应为 (num_windows, num_classes)

# 初始化 kFoldGenerator（数据已按窗口处理）
kfold_gen = kFoldGenerator(x_folds, y_folds)

# 初始化 DominGenerator（每个 fold 的窗口数量作为长度）
len_list = [x.shape[0] for x in x_folds]  # 每个 fold 生成的窗口数
domin_gen = DominGenerator(len_list)

batch_size = 128

for fold_idx in range(k):
    # 获取数据
    train_data, train_labels, val_data, val_labels = kfold_gen.getFold(fold_idx)
    train_domin, val_domin = domin_gen.getFold(fold_idx)
    
    # 输出形状验证
    print(f"\n=== Fold {fold_idx+1}/{k} ===")
    print("Train Data:", train_data.shape)      # (num_train_windows, window_size, features)
    print("Train Labels:", train_labels.shape)  # (num_train_windows, num_classes)
    print("Val Data:", val_data.shape)          # (num_val_windows, window_size, features)
    print("Val Labels:", val_labels.shape)      # (num_val_windows, num_classes)
    print("Train Domin Shape:", train_domin.shape)  # (num_train_windows, 9) 假设 9 个领域
    print("Val Domin Shape:", val_domin.shape)      # (num_val_windows, 9)

    # 转换为 TensorFlow Dataset
    import tensorflow as tf
    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_data, train_domin), train_labels)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((val_data, val_domin), val_labels)).batch(batch_size)
    
    # 在此处定义模型并训练
    # model.fit(train_dataset, validation_data=val_dataset, ...)
