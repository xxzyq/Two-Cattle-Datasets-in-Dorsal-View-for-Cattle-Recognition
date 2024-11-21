from torch.utils.data import Dataset, DataLoader
import os
import torch
import random
from PIL import Image
import numpy as np

# class CustomDataset(Dataset):
#     def __init__(self, directory, transform=None, mode='train'):
#         self.directory = directory
#         self.transform = transform
#         self.images = []
#         self.labels = []
#         self.days = []
#         pid_container = set()# 用于存储pid

#         for filename in os.listdir(directory):
#             if filename.endswith('.jpg') or filename.endswith('.png'):
#                 original_pid = int(filename.split('_')[0])
#                 pid_container.add(original_pid)# 添加pid到集合中
#                 if 'day' in filename:
#                     day = int(filename.split('day')[-1].split(".")[0])# 从文件名解析天数
#                 else:
#                     day = 0 # 如果没有天数信息，默认为0
#                 self.images.append(os.path.join(directory, filename))# 添加完整路径到图像列表
#                 self.days.append(day)# 添加天数到天数列表
#         # 生成从pid到标签的映射字典
#         pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
#         self.num_classes = len(pid2label)  # Store the number of unique labels

#         # 再次遍历目录，为每个图像分配标签
#         for filename in os.listdir(directory):
#             if filename.endswith('.jpg') or filename.endswith('.png'):
#                 original_pid = int(filename.split('_')[0])
#                 new_label = pid2label[original_pid] # 使用映射字典查找新的标签
#                 self.labels.append(new_label)   # 添加标签到标签列表
#         # 生成从标签到对应所有图像索引的映射字典
#         self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0] for label in set(self.labels)}

#     def __getitem__(self, index):
#         img_path = self.images[index]
#         label = self.labels[index]
#         day = self.days[index]
#         image = Image.open(img_path).convert('RGB')
#         if self.transform is not None:
#             image = self.transform(image)
#         return image, label, day

#     def __len__(self):
#         return len(self.images)
class CustomDataset(Dataset):
    def __init__(self, directory, transform=None, mode='train'):
        self.directory = directory
        self.transform = transform
        self.mode = mode  # 存储模式信息
        self.images = []
        self.labels = []
        self.days = []
        self.paths = []
        pid_container = set()

        # 遍历目录，收集图片路径、原始pid和日期信息
        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                extract_id = filename.split('_')[0]
                if extract_id.isdigit():
                    original_pid = int(extract_id)
                else:
                    original_pid = 0
                pid_container.add(original_pid)
                if 'day' in filename:
                    day = int(filename.split('day')[-1].split(".")[0])
                else:
                    part = filename.split('_')
                    date_parts1 = part[1].split("-")
                    date_parts2 = part[0].split("-")
                    if len(date_parts1) == 3:
                        year, month, day = date_parts1
                        day = int(year + month + day)  # 将年月日合并为一个整数
                    # part = filename.split('_')
                    # day = int(part[1].split("-")[0]+part[1].split("-")[1]+part[1].split("-")[2])
                    # if self.mode == 'val':
                    #     part = filename.split('_')
                    #     day = part[1]
                    elif len(date_parts2) == 3:
                        year, month, day = date_parts2
                        day = int(year + month + day)
                    else:
                        day = 0
                self.images.append(os.path.join(directory, filename))
                self.days.append(day)
                file_path = os.path.join(directory, filename)
                self.paths.append(file_path)
                # print(len(self.days),len(self.paths))
        if self.mode == 'train':
            # 如果是训练模式，对pid进行重新标注
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        else:
            # 如果是非训练模式，使用原始pid作为标签
            # pid2label = {pid: pid for pid in sorted(pid_container)}
            pid2label = {pid: label for label, pid in enumerate(sorted(pid_container))}
        self.num_classes = len(pid_container)  # 存储类别数目
        # 根据pid2label映射为每张图片赋予新的标签
        for filename in os.listdir(directory):
            if filename.endswith('.jpg') or filename.endswith('.png'):
                # original_pid = int(filename.split('_')[0])
                extract_id = filename.split('_')[0]
                if extract_id.isdigit():
                    original_pid = int(extract_id)
                else:
                    original_pid = 0
                new_label = pid2label[original_pid]
                self.labels.append(new_label)
        # 创建从标签到对应所有图像索引的映射
        self.label_to_indices = {label: np.where(np.array(self.labels) == label)[0]
                                 for label in set(self.labels)}
    def __getitem__(self, index):
        img_path = self.images[index]
        label = self.labels[index]
        day = self.days[index]
        path = self.paths[index]
        # print(len(self.days),len(self.paths))
        # print(day)
        image = Image.open(img_path).convert('RGB')
        if self.transform is not None:
            image = self.transform(image)
        return image, label, day, path
    def __len__(self):
        return len(self.images)
    
class CustomSampler(torch.utils.data.Sampler):
    def __init__(self, dataset, m):
        self.dataset = dataset  # 使用的数据集
        # self.n = n  # 每个批次中不同ID的数量
        self.m = m  # 每个ID在每个批次中的图像数量
        self.id_batches = self.make_batches()   # 创建批次

    def make_batches(self):
        batches = []    # 创建批次列表
        # 对每个标签的索引进行迭代
        for label_indices in self.dataset.label_to_indices.values():
            np.random.shuffle(label_indices)  # 打乱索引
            # 按每m个划分为一个批次
            for i in range(0, len(label_indices), self.m):
                batches.append(label_indices[i:i+self.m])

        np.random.shuffle(batches)  # Shuffle all batches to randomize the order of different labels
        return batches

    def __iter__(self):
        batch_counts = {i: 0 for i in range(len(self.id_batches))}  # Track usage of each batch
        used_batches = set()  # 跟踪已使用的批次
        # 循环直到所有批次都被使用
        while len(used_batches) < len(self.id_batches):
            idx = np.random.choice(len(self.id_batches))  # 随机选择一个批次索引
            if idx not in used_batches:
                yield from self.id_batches[idx]# 生成该批次的数据
                used_batches.add(idx)
                batch_counts[idx] += 1

    def __len__(self):
        # This is an estimate length based on average batch size times number of batches
        return len(self.id_batches) * self.m # 返回批次长度的估计值