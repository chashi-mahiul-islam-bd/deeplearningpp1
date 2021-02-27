#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 19 11:50:34 2021

@author: chashi
"""
import numpy as np
import torch
import numpy as np
from torch.utils.data import TensorDataset, DataLoader
import matplotlib.pyplot as plt
# T=[]
samples = []
labels = []
with open("zip_train.txt", "r") as file1:
    for line in file1.readlines():
        f_list = [float(i) for i in line.split(" ") if i!='\n']
        label = np.array([f_list[0]])
        sample = np.array([f_list[1:]])
        samples.append(sample)
        labels.append(label)


def load(batch_size=16, shuffle=True):
    tensor_x = torch.Tensor(samples)
    tensor_y = torch.Tensor(labels)
    dataset = TensorDataset(tensor_x, tensor_y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)
    return loader

if __name__ == "__main__":
    data_loader = load(batch_size = 16)
    
    di = iter(data_loader)
    
    s, i = di.next() 
    images, labels = di.next()
    images = images.numpy()
    images = images.reshape(16, 1 , 16, 16)
    
    plt.imshow(images[7].squeeze(), cmap='Greys_r')
    plt.title(str(labels[7].numpy()))    
