# -*- coding: utf-8 -*-
import os
import time
import torch.nn as nn
import numpy as np
import pandas as pd
from TSSTANet.tsstanet import stanet_af
import torch.utils.data
from dataloader.main_dataloader import ValDataset
from torch.utils.data import DataLoader

torch.multiprocessing.set_sharing_strategy('file_system')
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
DEVICE = torch.device('cuda')
SCORE_RANGE = 63
SAMPLE_INTERVAL = 3
frame_len = 64
features = 16
BATCHSIZE = 10


# Generate the model.
Net = stanet_af(layers=[2, 2, 2, 2], in_channels=3, num_classes=1, k=2, features=features)
Net.load_state_dict(torch.load(f'weights/best.pth', weights_only=True, map_location=DEVICE))
Net = Net.to(DEVICE)

# loss function
MSE_loss_func = nn.MSELoss()
MAE_loss_func = nn.L1Loss()

print('=== AVEC 2014 testing ===')

# Get the dataset.
# the `image_path_list` contains all image frames
df = pd.read_csv('./dataset/avec14.csv')
image_path_list = df['path'].values
label_list = df['label'].values
test_image_path_list = image_path_list[200:]
test_label_list = label_list[200:]

test_data = ValDataset(img_path=test_image_path_list, label_value=test_label_list, dataset='avec14',
                       frame_len=frame_len, img_size=224, input_channel=3,
                       sample_interval=SAMPLE_INTERVAL, transform=None)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=4, drop_last=False, pin_memory=True)

Net.eval()
RMSE_loss = []
MAE_loss = []
with torch.no_grad():
    for step, (test_img_pack, test_label) in enumerate(test_loader):
        predict_list = []
        for val_img_idx in range(0, test_img_pack.size(1), BATCHSIZE):
            predict = Net(test_img_pack[:,val_img_idx:val_img_idx+BATCHSIZE,:,:,:].to(DEVICE).squeeze(0))
            predict = torch.relu(predict) * SCORE_RANGE
            predict = predict.view(predict.size(0))
            predict_list.append(predict.mean().cpu())
        predict = torch.tensor(np.mean(predict_list)).unsqueeze(dim=0)  # mean value as final score of one video
        RMSE_loss.append(MSE_loss_func(predict, test_label))
        MAE_loss.append(MAE_loss_func(predict, test_label))
        # if (step + 1) % 10 == 0:
        #     print('Step: {:d} | val label: {:.4f} | val predict: {:.4f}'.format(
        #         step + 1, test_label.squeeze(), predict.squeeze()))

        ## old slow solution
        # for val_img in test_img_pack:
        #     predict = Net(val_img.to(DEVICE))
        #     predict = torch.relu(predict) * SCORE_RANGE
        #     predict = predict.view(predict.size(0))
        #     predict_list.append(predict.item())
        # predict = torch.tensor(np.mean(predict_list)).unsqueeze(dim=0)  # mean value as final score of one video
        # print(len(test_img_pack))
        # RMSE_loss.append(MSE_loss_func(predict, test_label))
        # MAE_loss.append(MAE_loss_func(predict, test_label))


    mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))
    mean_mae_loss = np.mean(MAE_loss)
    timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
    print('{} val MAE loss: {:.4f}    val RMSE loss: {:.4f}'.format(timestamp, mean_mae_loss, mean_rmse_loss))

torch.cuda.empty_cache()

print('=== AVEC 2013 testing ===')

# Get the dataset.
# the `image_path_list` contains all image frames
df = pd.read_csv('./dataset/avec13.csv')
image_path_list = df['path'].values
label_list = df['label'].values
test_image_path_list = image_path_list[100:]
test_label_list = label_list[100:]

test_data = ValDataset(img_path=test_image_path_list, label_value=test_label_list, dataset='avec13',
                       frame_len=frame_len, img_size=224, input_channel=3,
                       sample_interval=SAMPLE_INTERVAL, transform=None)
test_loader = DataLoader(test_data, batch_size=1, shuffle=False, num_workers=2, drop_last=False, pin_memory=True)

Net.eval()
RMSE_loss = []
MAE_loss = []
with torch.no_grad():
    for step, (test_img_pack, test_label) in enumerate(test_loader):
        predict_list = []
        for val_img_idx in range(0, test_img_pack.size(1), BATCHSIZE):
            predict = Net(test_img_pack[:, val_img_idx:val_img_idx + BATCHSIZE, :, :, :].to(DEVICE).squeeze(0))
            predict = torch.relu(predict) * SCORE_RANGE
            predict = predict.view(predict.size(0))
            predict_list.append(predict.mean().cpu())
        predict = torch.tensor(np.mean(predict_list)).unsqueeze(dim=0)  # mean value as final score of one video
        RMSE_loss.append(MSE_loss_func(predict, test_label))
        MAE_loss.append(MAE_loss_func(predict, test_label))
        # if (step + 1) % 10 == 0:
        #     print('Step: {:d} | val label: {:.4f} | val predict: {:.4f}'.format(
        #         step + 1, test_label.squeeze(), predict.squeeze()))

    mean_rmse_loss = np.sqrt(np.mean(RMSE_loss))
    mean_mae_loss = np.mean(MAE_loss)
    timestamp = time.strftime('%Y-%m-%d-%H_%M_%S', time.localtime(time.time()))
    print('{} val MAE loss: {:.4f}    val RMSE loss: {:.4f}'.format(timestamp, mean_mae_loss, mean_rmse_loss))

