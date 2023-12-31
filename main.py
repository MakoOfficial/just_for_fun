import torch
from torch import optim
from torch.optim.lr_scheduler import StepLR
from torch.optim.lr_scheduler import ReduceLROnPlateau
import random
import argparse

import timm
# assert timm.__version__ == "0.4.12" # version check

import models_mae
from model_utils.data import create_data_loader, split_data
from model_utils.train import train_VAL


def get_args_parser():
    """add the argument"""
    parser = argparse.ArgumentParser('MAE fine-tuning for image classification', add_help=False)

    #model
    parser.add_argument('--model', default='mae_vit_base_patch16', type=str, metavar='MODEL',
                       help='Name of model to train')
    #path
    parser.add_argument('--predModelPath', default='./mae_visualize_vit_base.pth',
                        help='finetune from checkpoint')

    return parser


def initMae(args):
    """load a pretrained-model"""
    model = models_mae.__dict__[args.model]()
    checkpoint = torch.load(args.predModelPath, map_location='cpu')
    load_state_dict = checkpoint['model']
    model_dict = model.state_dict()

    state_dict = {k: v for k, v in load_state_dict.items() if k in load_state_dict.keys()}
    model_dict.update(state_dict)
    model.load_state_dict(model_dict)
    return model

from models_mae import new_model
def initNewModel(backbone):
    """load a new model"""
    model = new_model(backbone)
    return model

def try_gpu(i=0):
    """如果存在，则返回gpu(i)，否则返回cpu()"""
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f'cuda:{i}')
    return torch.device('cpu')


#################################################################
random.seed(1)
batch_size = 128
learning_rate = 5e-3
num_epochs = 700
weight_decay = 1e-5
lr_period = 10
lr_decay = 0.8
epoch = 700
device = try_gpu()
##################################################################
savePath = './myPretrain_700epoch.pth'
##################################################################
bone_dir = "../archive"
csv_name = "boneage-training-dataset.csv"
train_df, valid_df = split_data(bone_dir, csv_name, 10, 0.05, 256)
train_loader, val_loader = create_data_loader(train_df, valid_df, batch_size)
##################################################################





if __name__ == '__main__':
    args = get_args_parser()
    args = args.parse_args()
    model = initMae(args).to(device)
    # model = initNewModel(initMae(args)).to(device)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    # scheduler = StepLR(optimizer, step_size=lr_period, gamma=lr_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10, threshold=1e-4)
    train_VAL(model=model, train_loader=train_loader, val_loader=val_loader, optimizer=optimizer, scheduler=scheduler,
              num_epoch=epoch, device=device, save_=savePath, record_path="./RECORD.csv")
