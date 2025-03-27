import os
import numpy as np
import torch
from torch.utils.data import Dataset

class TFs_Dataset(Dataset):
    def __init__(self, data_path, logger, select_class=None, len_time=1, size=512):
        """
        Initialization data file path and other data-related configurations 
        Read data from data file
        Preprocess the data
        """
        self.data_path = data_path
        self.logger = logger
        self.size = size
        self.time = int(len_time * self.size)

        if select_class is None:
            self.num_classes = 24   # 未指定类别，默认24类
            self.select_class = [i for i in range(24)]
        elif(max(select_class)>23):
            raise ValueError("选定的类别索引不能大于23")
        else:
            self.num_classes = len(select_class)    # 指定类别，则根据类别数量初始化
            self.select_class = select_class

        self.samples = self.__read_data()  # 初始化样本列表
        

    def __len__(self):
        """
        Dataset length
        """
        return len(self.samples)
    def __getitem__(self, index):
        """
        Return a set of data pairs (data[index], label[index])
        """
        path, label = self.samples[index]
        x = np.load(path)
        # resize feature into size 512
        x = x[0:self.time, :]
        x = x[:, 0:self.size]
        x = torch.FloatTensor(x)
        return x.unsqueeze(0), label
    
    def __read_data(self):
        samples = []
        for label, class_idx in enumerate(self.select_class):
            # 获取当前类别的目录路径
            class_path = os.path.join(self.data_path, str(class_idx))
            
            # 使用 os.scandir() 遍历目录中的所有文件
            with os.scandir(class_path) as entries:
                file_paths = [entry.path for entry in entries if entry.is_file()]  # 获取文件路径列表
            
            # 生成标签列表
            labels = [label] * len(file_paths)  # 使用乘法生成与文件数量相同的标签列表
            
            # 将文件路径和标签配对，添加到样本列表中
            samples.extend(zip(file_paths, labels))

        return samples
    def __preprocess_data(self):
        pass 
