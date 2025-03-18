import numpy as np
from DataGenerator import kFoldGenerator, DominGenerator

# 加载数据
data = np.load('ISRUC_S3_all.npz')
x_all = data['x']  # 假设shape=(num_subjects, num_samples, features)
y_all = data['y']  # 假设shape=(num_subjects, num_samples, num_classes)
k = x_all.shape[0]  # fold数（如10个受试者）

# 分割每个受试者的数据并生成窗口
window_size = 5
x_folds, y_folds = [], []
for i in range(k):
    # 提取第i个受试者的原始数据
    x_subject = x_all[i]  # (num_samples, features)
    y_subject = y_all[i]  # (num_samples, num_classes)
    
    # 生成时间窗口
    num_windows = x_subject.shape[0] - window_size + 1
    x_windows = np.array([x_subject[t:t+window_size] for t in range(num_windows)])
    y_windows = y_subject[window_size//2 : num_windows + window_size//2]  # 取窗口中心标签
    
    x_folds.append(x_windows)
    y_folds.append(y_windows)

# 初始化k折数据生成器
kfold_gen = kFoldGenerator(x_folds, y_folds)

# 初始化Domin生成器（每个fold的长度）
len_list = [len(fold) for fold in x_folds]
domin_gen = DominGenerator(len_list)

batch_size = 128

for fold_idx in range(k):
    # 获取数据
    train_data, train_labels, val_data, val_labels = kfold_gen.getFold(fold_idx)
    train_domin, val_domin = domin_gen.getFold(fold_idx)
    
    # 示例：输出数据形状
    print(f"Fold {fold_idx+1}/{k}")
    print("Train Data:", train_data.shape, "Labels:", train_labels.shape)
    print("Val Data:", val_data.shape, "Labels:", val_labels.shape)
    print("Train Domin:", train_domin.shape, "Val Domin:", val_domin.shape)
    
    # 转换为TensorFlow Dataset（示例）
    import tensorflow as tf
    train_dataset = tf.data.Dataset.from_tensor_slices(
        ((train_data, train_domin), train_labels)).batch(batch_size)
    val_dataset = tf.data.Dataset.from_tensor_slices(
        ((val_data, val_domin), val_labels)).batch(batch_size)
    
    # 在此处定义并训练模型
    # model.fit(train_dataset, validation_data=val_dataset, ...)

