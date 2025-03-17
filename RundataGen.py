import numpy as np
from DataGenerator import kFoldGenerator, DominGenerator

# 设置参数
batch_size = 128    # 论文中设置的 batch_size
window_size = 5     # 2T+1，T=2

# 加载数据
# 假设 npz 文件中包含 'x' 和 'y' 两个数组，请根据实际情况修改 key 名称
data = np.load('ISRUC_S3_all.npz')
x = data['x']
y = data['y']

# 如果 x 和 y 不是按照折数组织好的列表，需要先将数据划分为多个折（例如这里分为 5 折）
num_folds = 5
x_list = np.array_split(x, num_folds)
y_list = np.array_split(y, num_folds)

# 实例化 kFoldGenerator
generator = kFoldGenerator(x_list, y_list)

# 例如，获取第 0 折的训练和验证数据
train_data, train_targets, val_data, val_targets = generator.getFold(0)

# 打印数据形状，查看是否正确加载
print("训练数据大小:", train_data.shape)
print("训练标签大小:", train_targets.shape)
print("验证数据大小:", val_data.shape)
print("验证标签大小:", val_targets.shape)
