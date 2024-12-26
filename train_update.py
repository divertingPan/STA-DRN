# -*- coding: utf-8 -*-
import os
import time
import torch.nn as nn
import numpy as np
import pandas as pd
from TSSTANet.tsstanet import stanet_af
import torch.optim as optim
import torch.utils.data
from torch.amp import autocast, GradScaler
from dataloader.main_dataloader import MainDataset as Dataset
from dataloader.main_dataloader import ValDataset
from dataloader import transforms3d
from torch.utils.data import DataLoader
from torchvision.transforms import transforms


def distributed_label(true_labels, classes, sigma=0.0):
    with torch.no_grad():
        true_labels = true_labels.unsqueeze(dim=1)
        true_dist = torch.arange(0, classes, 1).repeat(true_labels.size(0), 1)
        true_dist = torch.true_divide(np.exp(torch.true_divide(-(true_dist - true_labels) ** 2,
                                                               (2 * sigma ** 2))),
                                      (sigma * np.sqrt(2 * np.pi)))
    return true_dist

torch.backends.cudnn.benchmark=True
torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device('cuda')
BATCHSIZE = 5
EPOCHS = 500
BACKPROP_STEP = 10
VAL_STEP = 10
SCORE_RANGE = 63
PRETRAIN = False
DATASET = 'avec14'
SAMPLE_INTERVAL = 3
optimizer_name = 'Adam'
lr = 0.000001
frame_len = 64
features = 16
sigma = 0

TAG = 'avec_publish'


if not os.path.exists(f'weights/{TAG}'):
    os.makedirs(f'weights/{TAG}')

# Generate the model.
Net = stanet_af(layers=[2, 2, 2, 2], in_channels=3, num_classes=1, k=2, features=features)
# Net = torch.nn.DataParallel(Net)
Net = Net.to(DEVICE)
if PRETRAIN:
    Net.load_state_dict(torch.load('weights/avec_all_train/100.pth', weights_only=True, map_location=DEVICE))

# Generate the optimizers.
optimizer = getattr(optim, optimizer_name)(Net.parameters(), lr=lr)
# scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, 50, 2)
scaler = GradScaler()
optimizer.zero_grad()

# loss function
MSE_loss_func = nn.MSELoss()
MAE_loss_func = nn.L1Loss()
Huber_loss_func = nn.SmoothL1Loss()

# Get the dataset.
df = pd.read_csv(f'./dataset/{DATASET}.csv')
image_path_list = df['path'].values
label_list = df['label'].values

if DATASET == 'avec14':
    train_size = 100
else:
    train_size = 50
train_image_path_list = image_path_list[:train_size]
train_label_list = label_list[:train_size]
val_image_path_list = image_path_list[train_size:2 * train_size]
val_label_list = label_list[train_size:2 * train_size]


train_transform = transforms.Compose([
    transforms3d.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms3d.RandomHorizontalFlip()
])
val_transform = None

train_data = Dataset(img_path=train_image_path_list, label_value=train_label_list, dataset=DATASET,
                     frame_len=frame_len, img_size=224, input_channel=3,
                     sample_interval=SAMPLE_INTERVAL, transform=train_transform)
train_loader = DataLoader(train_data, batch_size=BATCHSIZE, shuffle=True, num_workers=10,
                          drop_last=True, pin_memory=True, persistent_workers=True)
val_data = ValDataset(img_path=val_image_path_list, label_value=val_label_list, dataset=DATASET,
                      frame_len=frame_len, img_size=224, input_channel=3,
                      sample_interval=SAMPLE_INTERVAL, transform=val_transform)
val_loader = DataLoader(val_data, batch_size=1, shuffle=False, num_workers=10,
                        drop_last=False, pin_memory=True)

best_MAE = float('inf')
step_flag = 0
# Training of the model.
for epoch in range(EPOCHS):
    Net.train()
    RMSE_loss = []
    MAE_loss = []
    for step, (train_img, train_label) in enumerate(train_loader):
        step_flag += 1
        train_img = train_img.to(DEVICE)
        train_label = train_label + np.random.normal(0, sigma, train_label.shape[0])
        train_label = train_label.float().to(DEVICE) / SCORE_RANGE
        with autocast('cuda'):
            predict = Net(train_img)
            predict = predict.view(predict.size(0))
            loss = (MSE_loss_func(predict, train_label) + Huber_loss_func(predict, train_label)) / BACKPROP_STEP

        scaler.scale(loss).backward()
        if step_flag % BACKPROP_STEP == 0:
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        RMSE_loss.append(MSE_loss_func(predict * SCORE_RANGE, train_label * SCORE_RANGE).item())
        MAE_loss.append(MAE_loss_func(predict * SCORE_RANGE, train_label * SCORE_RANGE).item())
        mean_mae_loss = np.mean(MAE_loss)
        mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))

    # scheduler.step()

    print('Epoch: {:d}  Step: {:d} | '
          'train MAE loss: {:.4f}  RMSE loss: {:.4f} | LR: {:.6f}'.format(
        epoch+1, (train_size // BATCHSIZE),
        mean_mae_loss, mean_rmse_loss, optimizer.param_groups[0]['lr']))

    if (epoch + 1) % VAL_STEP == 0:
        Net.eval()
        RMSE_loss = []
        MAE_loss = []
        with torch.no_grad():
            for step, (val_img_pack, val_label) in enumerate(val_loader):
                predict_list = []
                for val_img_idx in range(0, val_img_pack.size(1), BATCHSIZE):
                    predict = Net(val_img_pack[:, val_img_idx:val_img_idx + BATCHSIZE, :, :, :].to(DEVICE).squeeze(0))
                    predict = torch.relu(predict) * SCORE_RANGE
                    predict = predict.view(predict.size(0))
                    predict_list.append(predict.mean().cpu())
                predict = torch.tensor(np.mean(predict_list)).unsqueeze(dim=0)  # mean value as final score of one video
                RMSE_loss.append(MSE_loss_func(predict, val_label))
                MAE_loss.append(MAE_loss_func(predict, val_label))
                # if (step + 1) % 10 == 0:
                #     print('Step: {:d} | val label: {:.4f} | val predict: {:.4f}'.format(
                #         step + 1, val_label.squeeze(), predict.squeeze()))

            mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))
            mean_mae_loss = np.mean(MAE_loss)
            timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
            print('{} val MAE loss: {:.4f}    val RMSE loss: {:.4f}'.format(timestamp, mean_mae_loss, mean_rmse_loss))

        torch.save(Net.state_dict(), f'./weights/{TAG}/{epoch + 1}.pth')
        if mean_mae_loss < best_MAE:
            best_MAE = mean_mae_loss
            print('Best MAE: {:.4f}, model saved!'.format(best_MAE))
