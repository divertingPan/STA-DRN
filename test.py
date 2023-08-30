# -*- coding: utf-8 -*-
import os
import time
import math
import torch.nn as nn
import numpy as np
import pandas as pd
from TSSTANet.tsstanet import tanet, sanet, stanet, stanet_af
import torch.optim as optim
import torch.utils.data
from dataloader.main_dataloader import MainDataset as Dataset
from dataloader.main_dataloader import ValDataset
from dataloader import transforms3d
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
import argparse


parser = argparse.ArgumentParser()
parser.add_argument("--sigma", default=0, type=int, help="sigma of gaussian noise")
parser.add_argument("--size", default=0, type=int, help="kernel size of gaussian blurring")
args = parser.parse_args()

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# DEVICE = 'cpu'
BATCHSIZE = 5
EPOCHS = 200
LOG_STEP = 10
SCORE_RANGE = 63


# params need to search
optimizer_name = 'Adam'
lr = 0.005
frame_len = 64
features = 16
sigma = 1

# Generate the model.
Net = stanet_af(layers=[2, 2, 2, 2], in_channels=3, num_classes=1, k=2, features=features)
Net.load_state_dict(torch.load('model/best.pth', map_location=DEVICE))
Net = Net.to(DEVICE)


# loss function
MSE_loss_func = nn.MSELoss()
MAE_loss_func = nn.L1Loss()
Huber_loss_func = nn.SmoothL1Loss()

# Get the dataset.
# the `image_path_list` contains all image frames
df = pd.read_csv('./datasets/avec14/label.csv')
image_path_list = df['path'].values
label_list = df['label'].values


test_image_path_list = image_path_list[200:300]
test_label_list = label_list[200:300]

# set this transform to None for normal testing
test_transform = transforms.Compose([
    transforms3d.Noise(sigma=args.sigma),  # [1, 2, 4, 8, 16, 32, 64, 128, 256]
    transforms3d.Blurring(size=args.size)  # [3, 5, 7, 15, 27, 49]
])

test_data = ValDataset(img_path=test_image_path_list, label_value=test_label_list, dataset='avec14',
                       frame_len=frame_len, img_size=224, input_channel=3, transform=test_transform)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=8)

Net.eval()
RMSE_loss = []
MAE_loss = []
with torch.no_grad():
    for step, (test_img_pack, test_label) in enumerate(test_loader):
        predict_list = []
        for val_img in test_img_pack:
            predict = Net(val_img.to(DEVICE))
            predict = torch.relu(predict) * SCORE_RANGE
            predict = predict.view(predict.size(0))
            predict_list.append(predict.item())
        predict = torch.tensor(np.mean(predict_list)).unsqueeze(dim=0)  # mean value as final score of one video
        RMSE_loss.append(MSE_loss_func(predict, test_label))
        MAE_loss.append(MAE_loss_func(predict, test_label))
        if (step + 1) % 10 == 0:
            print('Step: {:d} | val label: {:.4f} | val predict: {:.4f}'.format(
                step + 1, test_label.squeeze(), predict.squeeze()))

    mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))
    mean_mae_loss = np.mean(MAE_loss)
    timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
    print('{} val MAE loss: {:.4f}    val RMSE loss: {:.4f}'.format(timestamp, mean_mae_loss, mean_rmse_loss))
