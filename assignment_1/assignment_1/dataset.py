import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import Compose


class MnistDataset(Dataset):
    """This class processes the mnist dataset stored in file"""
    def __init__(self, file_path, test=False):
        """

        :param file_path: path of file
        :param test: specify whether the data is for training or testing/validation
        """
        self.image_target_tuple = self.image_target_tuple_getter(file_path)
        self.test = False

    def image_target_tuple_getter(self, file_path):
        image_target_tuple = []
        with open(file_path, "r") as ifile:
            for line in ifile.readlines():
                f_list = [float(i) for i in line.split(" ") if i != '\n']
                target = np.array([f_list[0]])
                image = np.array([f_list[1:]])
                image = np.reshape(image, (16, 16))
                image_target_tuple.append(image, target)
        return image_target_tuple

    def __len__(self):
        return len(self.image_target_tuple)

    def __getitem__(self, idx):
        """

        :param idx: tuple id
        :return:
        """
        sample = {}
        sample["image"], sample["target"] = self.image_target_tuple[idx]
        transforms = Compose([Reshape(), ToTensor()])
        for _, trans in enumerate(transforms):
            sample = trans(sample)
        return sample


class Reshape:
    """This is the reshape class which reshapes the data from 2d to 3d to pass the convolution"""
    def __call__(self, sample):
        """

        :param sample: dictionary containing image and label
        :return:
        """
        height, width = sample["images"].shape
        sample["image"] = np.reshape(sample["image"], (1, height, width))

class ToTensor:
    """This class will convert the images and targets to tensor"""
    def __call__(self, sample):
        """

        :param sample: dictionary containing image and label
        :return:
        """
        sample["image"] = torch.from_numpy(sample["image"])
        sample["target"] = torch.from_numpy(sample["target"]).long()