#!/bin/bash
# 创建用于 subgroup_1 的数据文件夹
mkdir -p ./data/ISRUC_S1/ExtractedChannels
mkdir -p ./data/ISRUC_S1/RawData
echo 'Make data dir: ./data/ISRUC_S1'

# 下载 raw data（注意这里的循环范围根据实际被试数量调整，比如假设有20个被试）
cd ./data/ISRUC_S1/RawData
for s in $(seq 1 100)  
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/subgroupI/$s.rar
    unrar x $s.rar
done
echo 'Download Data to "./data/ISRUC_S1/RawData" complete.'

# 下载 ExtractedChannels 数据
cd ../ExtractedChannels  
for s in $(seq 1 20)
do
    wget http://dataset.isr.uc.pt/ISRUC_Sleep/ExtractedChannels/subgroupI-Extractedchannels/subject$s.mat
done
echo 'Download ExtractedChannels to "./data/ISRUC_S1/ExtractedChannels" complete.'