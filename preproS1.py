import numpy as np
import scipy.io as scio
from os import path
from scipy import signal

# 配置路径
path_Extracted = './data/ISRUC_S1/ExtractedChannels/'
path_RawData = './data/ISRUC_S1/RawData/'
path_output = './data/ISRUC_S1/'
channels = ['C3_A2', 'C4_A1', 'F3_A2', 'F4_A1', 'O1_A2', 'O2_A1',
            'LOC_A2', 'ROC_A1', 'X1', 'X2']

def read_psg(path_Extracted, sub_id, channels, resample=3000):
    """读取并预处理PSG数据"""
    psg = scio.loadmat(path.join(path_Extracted, f'subject{sub_id}.mat'))
    psg_use = []
    for c in channels:
        # 重采样并添加通道维度
        channel_data = signal.resample(psg[c], resample, axis=-1)
        psg_use.append(np.expand_dims(channel_data, 1))
    return np.concatenate(psg_use, axis=1)  # 在通道维度拼接

def read_label(path_RawData, sub_id, ignore=30):
    """读取标签数据"""
    label = []
    label_path = path.join(path_RawData, f'{sub_id}/{sub_id}_1.txt')
    with open(label_path) as f:
        while (a := f.readline().strip()):
            label.append(int(a))
    return np.array(label[:-ignore])  # 忽略最后30个样本

# 初始化数据容器
fold_data = []
fold_label = []
fold_len = []

# 处理10个受试者
for sub_id in range(1, 11):
    print(f'Processing subject {sub_id}')
    
    # 读取数据
    labels = read_label(path_RawData, sub_id)
    psg_data = read_psg(path_Extracted, sub_id, channels)
    
    # 验证数据对齐
    assert len(labels) == len(psg_data), "数据与标签长度不匹配"
    print(f'Subject {sub_id} | Samples: {len(labels)} | PSG shape: {psg_data.shape}')

    # 标签映射：5->4 (REM类别)
    labels[labels == 5] = 4
    fold_label.append(np.eye(5)[labels])  # One-hot编码
    fold_data.append(psg_data)
    fold_len.append(len(labels))

# 保存数据（关键修改部分）
np.savez(
    path.join(path_output, 'ISRUC_S1_all.npz'),
    Fold_data=np.array(fold_data, dtype=object),    # 转换为对象数组
    Fold_label=np.array(fold_label, dtype=object),   # 允许不同形状
    Fold_len=np.array(fold_len)                      # 一维数组
)

print(f'Data saved to {path.join(path_output, "ISRUC_S1_all.npz")}')
