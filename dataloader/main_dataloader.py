"""
This dataset method generates a series of aligned face frames
"""

import re
import os
import torch
import numpy as np
from PIL import Image
from math import ceil
from torch.utils.data import Dataset
from dataloader import transforms3d


def smooth_one_hot(true_labels, classes, smoothing=0.0):
    """
    if smoothing == 0, it's one-hot method
    if 0 < smoothing < 1, it's smooth method
    ref: https://blog.csdn.net/sinat_38685124/article/details/108382216
    """
    assert 0 <= smoothing < 1
    confidence = 1.0 - smoothing
    label_shape = torch.Size((true_labels.size(0), classes))  # torch.Size([2, 5])
    with torch.no_grad():
        true_dist = torch.empty(size=label_shape, device=true_labels.device)  # 空的，没有初始化
        true_dist.fill_(smoothing / (classes - 1))
        index = true_labels
        true_dist.scatter_(1, torch.LongTensor(index.unsqueeze(1)), confidence)  # 必须要torch.LongTensor()
    return true_dist


class MainDataset(Dataset):

    def __init__(self, img_path, label_value, dataset,
                 frame_len=16, img_size=112, input_channel=3, sample_interval=3, transform=None):
        self.img_path = img_path
        self.label_value = label_value
        self.frame_len = frame_len
        self.img_size = img_size
        self.transform = transform
        self.input_channel = input_channel
        self.dataset = dataset
        self.sample_interval = sample_interval

    def __len__(self):
        return len(self.label_value)

    def __getitem__(self, idx):

        label = self.label_value[idx]

        if self.dataset == 'avec14':
            image_path = os.path.join('dataset', 'avec14', 'image', self.img_path[idx].replace('.mp4', '_aligned'))
        elif self.dataset == 'avec13':
            image_path = os.path.join('dataset', 'avec13', 'image', self.img_path[idx].replace('.mp4', '_aligned'))
        elif self.dataset == 'ucf101':
            image_path = os.path.join('dataset', 'ucf101', 'ucf101_jpegs_256', self.img_path[idx])
        elif self.dataset == 'ck+':
            image_path = os.path.join('dataset', 'CK+', 'image_cropped', self.img_path[idx])
        else:
            image_path = os.path.join('dataset', 'YouTube_Faces', 'aligned_cropped', self.img_path[idx])

        image_name = os.listdir(image_path)
        # if self.dataset == 'avec14':
        #     image_name.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        # else:
        #     image_name = sorted(image_name, key=str)
        image_name = sorted(image_name, key=str)
        image_name = image_name[::self.sample_interval]

        if len(image_name) <= self.frame_len:
            frames = [(i * len(image_name) // self.frame_len) for i in range(self.frame_len)]
        else:
            # 1. use same sample interval, random position
            rand_id = np.random.randint(0, len(image_name) - self.frame_len)
            frames = [i for i in range(rand_id, rand_id + self.frame_len)]

            # # 2. use full video clip, variant interval
            # frames = [(i * len(image_name) // self.frame_len) for i in range(self.frame_len)]

        image_pack = np.empty((self.frame_len, self.img_size, self.img_size, 3), np.dtype('float32'))

        for i, frame in enumerate(frames):
            image = Image.open(os.path.join(image_path, image_name[frame]))
            image = image.resize((self.img_size, self.img_size))
            image_pack[i] = np.asarray(image)

        image = np.transpose(image_pack, (3, 0, 1, 2))

        if self.transform:
            image = self.transform(image)

        if self.input_channel == 1:
            image = transforms3d.rgb_to_gray(image)
            # image = transforms3d.histeq(image)

        image = transforms3d.to_tensor(image)

        # image standardized
        image = transforms3d.normalize(image)

        return image, label


class ValDataset(Dataset):

    def __init__(self, img_path, label_value, dataset,
                 frame_len=16, img_size=112, input_channel=3, sample_interval=3, transform=None):
        self.img_path = img_path
        self.label_value = label_value
        self.frame_len = frame_len
        self.img_size = img_size
        self.transform = transform
        self.input_channel = input_channel
        self.dataset = dataset
        self.sample_interval = sample_interval

    def __len__(self):
        return len(self.label_value)

    def __getitem__(self, idx):

        label = self.label_value[idx]

        if self.dataset == 'avec14':
            image_path = os.path.join('dataset', 'avec14', 'image', self.img_path[idx].replace('.mp4', '_aligned'))
        elif self.dataset == 'avec13':
            image_path = os.path.join('dataset', 'avec13', 'image', self.img_path[idx].replace('.mp4', '_aligned'))
        elif self.dataset == 'ucf101':
            image_path = os.path.join('dataset', 'ucf101', 'ucf101_jpegs_256', self.img_path[idx])
        elif self.dataset == 'ck+':
            image_path = os.path.join('dataset', 'CK+', 'image_cropped', self.img_path[idx])
        else:
            image_path = os.path.join('dataset', 'YouTube_Faces', 'aligned_cropped', self.img_path[idx])

        image_name = os.listdir(image_path)
        # if self.dataset == 'avec14':
        #     image_name.sort(key=lambda i: int(re.match(r'(\d+)', i).group()))
        # else:
        #     image_name = sorted(image_name, key=str)
        image_name = sorted(image_name, key=str)
        image_name = image_name[::self.sample_interval]

        if len(image_name) <= self.frame_len:
            frames = []
            frames.append([(i * len(image_name) // self.frame_len) for i in range(self.frame_len)])
        else:

            ''' 
            Todo: add the cut up of a video
                1 2 3 4 5 6 7 8 9 10 11 12 13 (14 15)
                cut into parts with 3 elements (i.e. frames to input the network),
                firstly, 4 parts can fully spread the list,
                if add to 5 parts, some elements (14 15) will be missed
                
                should be split into 5 parts, every slice begins from i * (3 - ((5 * 3 - len()) / 5 - 1))
                How to get the value 5? 
                int(len() / 3)
                i.e., 
                from math import ceil
                li = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13]
                frame_len = 6
                section = ceil(len(li) / frame_len)
                slice_idx = [int(i * (frame_len - ((section * frame_len - len(li)) / int(section - 1))))
                             for i in range(section)]
                
            '''
            section = ceil(len(image_name) / self.frame_len)
            slice_idx = [int(i * (self.frame_len - ((section * self.frame_len - len(image_name)) / int(section - 1))))
                         for i in range(section)]
            frames = []
            for idx in slice_idx:
                frames.append([i for i in range(idx, idx + self.frame_len)])

        image_pack = np.empty((self.frame_len, self.img_size, self.img_size, 3), np.dtype('float32'))

        val_image_array = torch.empty((len(frames), 3, self.frame_len, self.img_size, self.img_size))
        if self.input_channel == 1:
            val_image_array = torch.empty((len(frames), 1, self.frame_len, self.img_size, self.img_size))
        # val_image_array = []
        for idx, section_frames in enumerate(frames):
            for i, frame in enumerate(section_frames):
                image = Image.open(os.path.join(image_path, image_name[frame]))
                image = image.resize((self.img_size, self.img_size))
                image_pack[i] = np.asarray(image)

            image = np.transpose(image_pack, (3, 0, 1, 2))

            if self.transform:
                image = self.transform(image)

            if self.input_channel == 1:
                image = transforms3d.rgb_to_gray(image)
                # image = transforms3d.histeq(image)

            image = transforms3d.to_tensor(image)

            # image standardized
            image = transforms3d.normalize(image)

            # val_image_array.append(image)
            val_image_array[idx] = image

        return val_image_array, label

