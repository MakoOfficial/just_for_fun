import torch
import matplotlib.pyplot as plt
import time
import numpy as np
import torch.nn as nn
import torch.nn.init as init
from torch.utils.data import DataLoader, Dataset
import csv

loss_fn = nn.MSELoss()

def train_VAL(model, train_loader, val_loader, optimizer, scheduler, num_epoch, device, save_="model.pth",
              record_path="./RECORD.csv"):
    # 用测试集训练模型model(),用验证集作为测试集来验证
    record = [['epoch', 'training loss', 'val loss', 'lr']]
    with open(record_path, 'w', newline='') as csvfile:
        writer = csv.writer(csvfile)
        for row in record:
            writer.writerow(row)

    plt_train_loss = []
    plt_val_loss = []

    seed = 101
    torch.manual_seed(seed)

    for epoch in range(num_epoch):
        # update_lr(optimizer,epoch)
        print(f"MAE TRAINING!!\nEpoch:{epoch + 1}")
        record = []

        epoch_start_time = time.time()
        train_loss = 0.0
        val_loss = 0.0

        model.train()
        for i, data in enumerate(train_loader):
            optimizer.zero_grad()
            image = data.type(torch.FloatTensor).to(device)
            _, pred, mask = model(image, 0.75)
            pred = model.unpatchify(pred)
            loss = loss_fn(pred, image)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        #验证集val
        model.eval()

        with torch.no_grad():
            for i, data in enumerate(val_loader):
                image = data.type(torch.FloatTensor).to(device)
                loss, pred, mask = model(image, 0.75)

                val_loss += loss.item()

            plt_train_loss.append(train_loss/train_loader.dataset.__len__())
            plt_val_loss.append(val_loss/val_loader.dataset.__len__())

            record.append([epoch, train_loss/train_loader.dataset.__len__(), val_loss/val_loader.dataset.__len__(),
                           optimizer.param_groups[0]["lr"]])
            #将结果 print 出來
            print('[%03d/%03d] %22.f sec(s) Train Loss: %3.6f | Val loss: %3.6f' % \
                  (epoch + 1, num_epoch, time.time()-epoch_start_time, plt_train_loss[-1], plt_val_loss[-1]))

        scheduler.step()
        with open(record_path, 'a+', newline='') as csvfile:
            writer = csv.writer(csvfile)
            for row in record:
                writer.writerow(row)
        if epoch == num_epoch-1:
            torch.save(model, save_)

    # Loss曲线
    plt.plot(plt_train_loss)
    plt.plot(plt_val_loss)
    plt.title('Loss')
    plt.legend(['train', 'val'])
    plt.savefig('loss.png')
    plt.show()
