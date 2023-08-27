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


torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
# DEVICE = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
DEVICE = 'cpu'
BATCHSIZE = 5
EPOCHS = 200
LOG_STEP = 10
SCORE_RANGE = 63

if not (os.path.exists('model')):
    os.makedirs('model')

# params need to search
optimizer_name = 'Adam'
lr = 0.005
frame_len = 64
features = 16
sigma = 1

# Generate the model.
Net = stanet_af(layers=[2, 2, 2, 2], in_channels=3, num_classes=1, k=2, features=features)
Net = torch.nn.DataParallel(Net)
Net = Net.to(DEVICE)

# Generate the optimizers.
optimizer = getattr(optim, optimizer_name)(Net.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 2)

# loss function
MSE_loss_func = nn.MSELoss()
MAE_loss_func = nn.L1Loss()
Huber_loss_func = nn.SmoothL1Loss()

# Get the dataset.
# the `image_path_list` contains all image frames
df = pd.read_csv('./datasets/avec14/label.csv')
image_path_list = df['path'].values
label_list = df['label'].values

train_image_path_list = image_path_list[:100]
train_label_list = label_list[:100]
val_image_path_list = image_path_list[100:200]
val_label_list = label_list[100:200]

train_transform = transforms.Compose([
    transforms3d.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms3d.RandomHorizontalFlip()
])
val_transform = None

train_data = Dataset(img_path=train_image_path_list, label_value=train_label_list, dataset='avec14',
                     frame_len=frame_len, img_size=224, input_channel=3, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True, num_workers=8)
val_data = ValDataset(img_path=val_image_path_list, label_value=val_label_list, dataset='avec14',
                      frame_len=frame_len, img_size=224, input_channel=3, transform=val_transform)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=8)

best_MAE = float('inf')
total_step = math.ceil(100 / BATCHSIZE)
# Training of the model.
for epoch in range(EPOCHS):
    Net.train()
    RMSE_loss = []
    MAE_loss = []
    for step, (train_img, train_label) in enumerate(train_loader):
        predict = Net(train_img.to(DEVICE))
        predict = predict * SCORE_RANGE
        predict = predict.view(predict.size(0))

        train_label = train_label + np.random.normal(0, sigma, train_label.shape[0])
        train_label = train_label.float().to(DEVICE)

        loss = MSE_loss_func(predict, train_label)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        RMSE_loss.append(MSE_loss_func(predict, train_label).item())
        MAE_loss.append(MAE_loss_func(predict, train_label).item())
        mean_mae_loss = np.mean(MAE_loss)
        mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))

        if (step+1) % LOG_STEP == 0:
            print('Epoch: {:d}  Step: {:d} / {:d} | '
                  'train MAE loss: {:.4f} | train RMSE loss: {:.4f} | LR: {:.6f}'.format(
                   epoch, step+1, total_step,
                   mean_mae_loss, mean_rmse_loss, optimizer.param_groups[0]['lr']))

    scheduler.step()

    Net.eval()
    RMSE_loss = []
    MAE_loss = []
    with torch.no_grad():
        for step, (val_img_pack, val_label) in enumerate(val_loader):
            predict_list = []
            for val_img in val_img_pack:
                predict = Net(val_img.to(DEVICE))
                predict = torch.relu(predict) * SCORE_RANGE
                predict = predict.view(predict.size(0))
                predict_list.append(predict.item())
            predict = torch.tensor(np.mean(predict_list)).unsqueeze(dim=0)  # mean value as final score of one video
            RMSE_loss.append(MSE_loss_func(predict, val_label))
            MAE_loss.append(MAE_loss_func(predict, val_label))
            if (step + 1) % 10 == 0:
                print('Step: {:d} | val label: {:.4f} | val predict: {:.4f}'.format(
                    step + 1, val_label.squeeze(), predict.squeeze()))

        mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))
        mean_mae_loss = np.mean(MAE_loss)
        timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
        print('{} val MAE loss: {:.4f}    val RMSE loss: {:.4f}'.format(timestamp, mean_mae_loss, mean_rmse_loss))

    if mean_mae_loss < best_MAE:
        best_MAE = mean_mae_loss
        torch.save(Net.module.state_dict(), './model/{}.pth'.format('param_train_stanet_af'))
        print('Best MAE: {:.4f}, model saved!'.format(best_MAE))
