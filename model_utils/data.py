import cv2
import os
import numpy as np
import torchvision.transforms as transforms
import torchvision
from torchvision import transforms,datasets
from torch.utils.data import DataLoader, Dataset, random_split
from sklearn.model_selection import train_test_split
import torch
import random
from imblearn.over_sampling import SMOTE
from collections import Counter



HW = 224

def readjpgfile(listpath,label,rate = None):
    assert rate == None or rate//1 == rate
    # label 是一个布尔值，代表需不需要返回 y 值
    image_dir = sorted(os.listdir(listpath))
    n = len(image_dir)
    if rate:
        n = n*rate
    # x存储图片，每张彩色图片都是128(高)*128(宽)*3(彩色三通道)
    x = np.zeros((n, HW , HW , 3), dtype=np.uint8)
    # y存储标签，每个y大小为1
    y = np.zeros(n, dtype=np.uint8)
    if not rate:
        for i, file in enumerate(image_dir):
            img = cv2.imread(os.path.join(listpath, file))
            # xshape = img.shape
            # Xmid = img.shape[1]//2
            # 利用cv2.resize()函数将不同大小的图片统一为128(高)*128(宽) os.path.join作用是将两个路径拼接起来。路径+文件名
            x[i, :, :] = cv2.resize(img,(HW , HW ))
            y[i] = label
    else:
        for i, file in enumerate(image_dir):
            img = cv2.imread(os.path.join(listpath, file))
            # xshape = img.shape
            # Xmid = img.shape[1]//2
            # 利用cv2.resize()函数将不同大小的图片统一为128(高)*128(宽) os.path.join作用是将两个路径拼接起来。路径+文件名
            for j in range(rate):
                x[rate*i + j, :, :] = cv2.resize(img,(HW , HW ))
                y[rate*i + j] = label

    return x,y


#training 时，通过随机旋转、水平翻转图片来进行数据增强（data_abnor augmentation）
train_transform = transforms.Compose([
    # transforms.RandomResizedCrop(150),
    transforms.ToPILImage(),
    transforms.ToTensor()
    # transforms.Normalize(mean=[0.485, 0.456, 0.406],
    #                      std=[0.229, 0.224, 0.225]
])
 
#testing 时，不需要进行数据增强（data_abnor augmentation）
test_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.ToTensor(),
])
 
class ImgDataset(Dataset):
 
    def __init__(self, x, y=None, transform=None, lessTran = False):
        self.x = x
        # label 需要是 LongTensor 型
        self.y = y
        if y is not None:
            self.y = torch.LongTensor(y)
        self.transform = transform
        self.lessTran = lessTran
        # 强制水平翻转
        self.trans0 = torchvision.transforms.Compose([
            transforms.ToPILImage(),
            torchvision.transforms.Resize(256),
                                                      torchvision.transforms.RandomCrop(224),
                                                      torchvision.transforms.RandomHorizontalFlip(p=1),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                       [0.229, 0.224, 0.225])
                                                      ])
        # 强制垂直翻转
        self.trans1 = torchvision.transforms.Compose([
            transforms.ToPILImage(),
            torchvision.transforms.Resize(256),
                                                      torchvision.transforms.RandomCrop(224),
                                                      torchvision.transforms.RandomVerticalFlip(p=1),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                       [0.229, 0.224, 0.225])
                                                      ])
        # 旋转-90~90
        self.trans2 = torchvision.transforms.Compose([
            transforms.ToPILImage(),torchvision.transforms.Resize(256),
                                                      torchvision.transforms.RandomCrop(224),
                                                      torchvision.transforms.RandomRotation(90),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                       [0.229, 0.224, 0.225])
                                                      ])
 
        # 亮度在0-2之间增强，0是原图
        self.trans3 = torchvision.transforms.Compose([
            transforms.ToPILImage(),torchvision.transforms.Resize(256),
                                                      torchvision.transforms.RandomCrop(224),
                                                      torchvision.transforms.ColorJitter(brightness=1),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                       [0.229, 0.224, 0.225])
                                                      ])
        # 修改对比度，0-2之间增强，0是原图
        self.trans4 = torchvision.transforms.Compose([
            transforms.ToPILImage(),torchvision.transforms.Resize(256),
                                                      torchvision.transforms.RandomCrop(224),
                                                      torchvision.transforms.ColorJitter(contrast=2),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                       [0.229, 0.224, 0.225])
                                                      ])
        # 颜色变化
        self.trans5 = torchvision.transforms.Compose([
            transforms.ToPILImage(),torchvision.transforms.Resize(256),
                                                      torchvision.transforms.RandomCrop(224),
                                                      torchvision.transforms.ColorJitter(hue=0.5),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                       [0.229, 0.224, 0.225])
                                                      ])
        # 混合
        self.trans6 = torchvision.transforms.Compose([
            transforms.ToPILImage(),torchvision.transforms.Resize(256),
                                                      torchvision.transforms.RandomCrop(224),
                                                      torchvision.transforms.ColorJitter(brightness=1, contrast=2, hue=0.5),
                                                      torchvision.transforms.ToTensor(),
                                                      torchvision.transforms.Normalize([0.485, 0.456, 0.406],
                                                                                       [0.229, 0.224, 0.225])
                                                      ])
        self.trans_list = [self.trans0, self.trans1, self.trans2, self.trans3, self.trans4, self.trans5, self.trans6]
 
 
 
 
 
    def __len__(self):
        return len(self.x)
 
    def __getitem__(self, index):
        X = self.x[index]
 
        if self.y is not None:
            if  self.lessTran:
                num = random.randint(0, 6)
                X = self.trans_list[num](X)
            else:
                if self.transform is not None:
                    X = self.transform(X)
            Y = self.y[index]
            return X, Y
        else:
            return X
    def getbatch(self,indices):
        images = []
        labels = []
        for index in indices:
            image,label = self.__getitem__(index)
            images.append(image)
            labels.append(label)
        return torch.stack(images),torch.tensor(labels)
 
 
 
def getDateset(dir_class1, dir_class2, testSize=0.3,rate = None, testNum = None, lessTran = False):
    '''
    :param dir_class1:   这个是参数较少的那个
    :param dir_class2:
    :param testSize:
    :param rate:
    :param testNum:
    :return:
    '''
    x1,y1 = readjpgfile(dir_class1,0,rate=rate)  #类1是0
    x2,y2 = readjpgfile(dir_class2,1)   #类2是1
    if testNum == -1:
        X = np.concatenate((x1, x2))
        Y = np.concatenate((y1, y2))
        dataset = ImgDataset(X, Y, transform=train_transform, lessTran = lessTran)
        return dataset
    if not testNum :
        X = np.concatenate((x1, x2))
        Y = np.concatenate((y1, y2))
        train_x, test_x, train_y, test_y = train_test_split(X,Y,test_size=testSize,random_state=0)
 
    else:
        train_x1, test_x1, train_y1, test_y1 = train_test_split(x1,y1,test_size=testNum/len(y1),random_state=0)
        train_x2, test_x2, train_y2, test_y2 = train_test_split(x2,y2,test_size=testNum/len(y2),random_state=0)
        print(len(test_y2),len(test_y1))
        train_x = np.concatenate((train_x1,train_x2))
        test_x = np.concatenate((test_x1, test_x2))
        train_y = np.concatenate((train_y1,train_y2))
        test_y = np.concatenate((test_y1, test_y2))
 
    train_dataset = ImgDataset(train_x,train_y ,transform=train_transform,lessTran = lessTran)
    test_dataset = ImgDataset(test_x ,test_y,transform=test_transform,lessTran = lessTran)
 
    # test_x1,test_y1 = readjpgfile(r'F:\li_XIANGMU\pycharm\deeplearning\cat_dog\catsdogs\test\Cat',0)  #猫是0
    # test_x2,test_y2 = readjpgfile(r'F:\li_XIANGMU\pycharm\deeplearning\cat_dog\catsdogs\test\Dog',1)
    # test_x = np.concatenate((test_x1,test_x2))
    # test_y = np.concatenate((test_y1,test_y2))
 
 
    return train_dataset, test_dataset
 
 
 
def smote(X_train,y_train):
    oversampler = SMOTE(sampling_strategy='auto', random_state=np.random.randint(100), k_neighbors=5, n_jobs=-1)
    os_X_train, os_y_train = oversampler.fit_resample(X_train,y_train)
    print('Resampled dataset shape {}'.format(Counter(os_y_train)))
    return os_X_train, os_y_train
 
 
def getDataLoader(class1path, class2path, batchSize,mode='train'):
    assert mode in ['train','val', 'test']
    if mode == 'train':
        train_set = getDateset(class1path, class2path, testNum=-1)
 
        trainloader = DataLoader(train_set,batch_size=batchSize, shuffle=True)
 
        return trainloader
 
 
    elif mode == 'test':
        testset = getDateset(class1path, class2path, testNum=-1)
        testLoader = DataLoader(testset, batch_size=1, shuffle=False)
        return testLoader
imagenet_mean = np.array([0.485, 0.456, 0.406])
imagenet_std = np.array([0.229, 0.224, 0.225])

transform = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])
from PIL import Image


def read_image(file_path, image_size=224):
    """读取图片，并统一修改为512x512"""
    img = Image.open(file_path)
    img = img.resize((image_size, image_size))
    img = np.array(img.convert("RGB")) / 255.
    img = img - imagenet_mean
    img = img / imagenet_std

    return img

import pandas as pd

def split_data(data_dir, csv_name, category_num, split_ratio, aug_num):
    """重构数据级结构"""
    age_df = pd.read_csv(os.path.join(data_dir, csv_name))
    age_df['path'] = age_df['id'].map(lambda x: os.path.join(data_dir,
                                                             csv_name.split('.')[0],
                                                             '{}.png'.format(x)))
    age_df['exists'] = age_df['path'].map(os.path.exists)
    print(age_df['exists'].sum(), 'images found of', age_df.shape[0], 'total')

    age_df.dropna(inplace=True)
    age_df['boneage_category'] = pd.cut(age_df['boneage'], category_num)

    raw_train_df, valid_df = train_test_split(
        age_df,
        test_size=split_ratio,
        random_state=2023,
        stratify=age_df['boneage_category']
    )
    print('train', raw_train_df.shape[0], 'validation', valid_df.shape[0])
    train_df = raw_train_df.groupby(['boneage_category']).apply(lambda x: x.sample(aug_num, replace=True)).reset_index(drop=True)
    print('New Data Size:', train_df.shape[0], 'Old Size:', raw_train_df.shape[0])
    # raw_train_df.to_csv("train.csv")
    train_df.to_csv("train.csv")
    valid_df.to_csv("valid.csv")
    # return raw_train_df, valid_df
    return train_df, valid_df


# create 'dataset's subclass,we can read a picture when we need in training trough this way
import torch.utils.data.dataset as Dataset

class BAADataset(Dataset.Dataset):
    """重写Dataset类"""

    def __init__(self, df) -> None:
        self.df = df

    def __getitem__(self, index):
        row = self.df.iloc[index]
        return transform(read_image(row["path"], image_size=224))

    def __len__(self):
        return len(self.df)


def create_data_loader(train_df, val_df, batch_size):
    train_dataset, valid_dataset = BAADataset(train_df), BAADataset(val_df)
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=6,
        drop_last=True)

    val_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        drop_last=True,
        num_workers=6,
        shuffle=True)
    return train_loader, val_loader