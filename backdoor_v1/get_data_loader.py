from torchvision import datasets
from torchvision import transforms
from torch.utils.data.dataloader import DataLoader
from torch.optim.lr_scheduler import LambdaLR
import torch.nn.functional as F

import torch
import math
import argparse
import tabulate
import time
import numpy as np
from PIL import Image
import copy

def Remove_and_Reconsitution(img):
    f = np.fft.fft2(img)
    fshift = np.fft.fftshift(f) ## shift for centering 0.0 (x,y)

    rows = np.size(img, 0) #taking the size of the image
    cols = np.size(img, 1)
    crow, ccol = rows//2, cols//2

    fshift[crow-10:crow+10, ccol-10:ccol+10] = 0
    f_ishift= np.fft.ifftshift(fshift)

    img_back = np.fft.ifft2(f_ishift) ## shift for centering 0.0 (x,y)
    img_back = np.abs(img_back)

    return img_back

def Change_and_Reconsitution(base_img, high_freq_img,percentage):
    f = np.fft.fft2(base_img)
    fshift = np.fft.fftshift(f) ## shift for centering 0.0 (x,y)

    f_high_freq_img = np.fft.fft2(high_freq_img)
    fshift_high_freq_img = np.fft.fftshift(f_high_freq_img) ## shift for centering 0.0 (x,y)

    rows = np.size(base_img, 0) #taking the size of the image
    cols = np.size(base_img, 1)
    crow, ccol = rows//2, cols//2

    cut_size = int((1-percentage)*crow)

    fshift_high_freq_img[crow-cut_size:crow+cut_size, ccol-cut_size:ccol+cut_size] = copy.deepcopy(fshift[crow-cut_size:crow+cut_size, ccol-cut_size:ccol+cut_size])
    f_ishift= np.fft.ifftshift(fshift_high_freq_img)

    img_back = np.fft.ifft2(f_ishift) ## shift for centering 0.0 (x,y)
    img_back = np.abs(img_back)

    return img_back

def get_expend_dataset(args, num_expand_x, transform, index):
    train_labeled_idxs = x_expend(num_expand_x, index)
    train_labeled_dataset = ArdisDataset(
        args, train=True, transform=transform, indexs=train_labeled_idxs)

    return train_labeled_dataset

def x_expend(num_expand_x, index):
    labeled_idx = copy.deepcopy(index)
    exapand_labeled = num_expand_x // len(labeled_idx)
    labeled_idx = np.hstack(
        [labeled_idx for _ in range(exapand_labeled)])
    if len(labeled_idx) < num_expand_x:
        diff = num_expand_x - len(labeled_idx)
        labeled_idx = np.hstack(
            (labeled_idx, np.random.choice(labeled_idx, diff)))
    else:
        assert len(labeled_idx) == num_expand_x

    return labeled_idx

class ArdisDataset(torch.utils.data.Dataset):
    def __init__(self, args, transform = None, train = True, indexs=None):

        if train:
            X = np.loadtxt('../data/ARDIS_DATASET_IV/ARDIS_train_2828.csv', dtype='uint8')
            Y = np.loadtxt('../data/ARDIS_DATASET_IV/ARDIS_train_labels.csv', dtype='float')
        else:
            X = np.loadtxt('../data/ARDIS_DATASET_IV/ARDIS_test_2828.csv', dtype='uint8')
            Y = np.loadtxt('../data/ARDIS_DATASET_IV/ARDIS_test_labels.csv', dtype='float')

        Y = np.argmax(Y,axis = 1)
        X = X[Y==7]
        self.X  = X
        if not indexs is None:
            self.X = self.X[indexs]

        self.transform = transform
        self.attack_target = args.target

        self.attack_type = args.trigger_type

    def __len__(self):
        return len(self.X)

    def __getitem__(self,index):
        img = self.X[index]
        img = np.reshape(img, (28,28))

        if self.attack_type == 'edge':
            img = np.array(img,dtype='uint8')
            img = Image.fromarray(img)
            # img.save(f"Pic_edge.jpeg")
        if self.attack_type == 'edge_high_freq':
            img = Remove_and_Reconsitution(img)
            img = np.array(img,dtype='uint8')
            img = Image.fromarray(img)
            # img.save(f"Pic_edge_high_freq.jpeg")
        if self.attack_type == 'edge_low_freq':
            img1 = Remove_and_Reconsitution(img)
            img = np.abs(img - img1)
            img = np.array(img,dtype='uint8')
            img = Image.fromarray(img)
            # img.save(f"Pic_edge_low_freq.jpeg")

        target = int(self.attack_target)

        if self.transform is not None:
            img = self.transform(img)
            # print(img.max(), img.min())

        return img, target

def get_ardis(args, len_train_dataset, transform):
    root='../data'

    transform_labeled = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomCrop(size=28,
                              padding=int(28*0.125),
                              padding_mode='reflect'),
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    transform_val = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    train_dataset = ArdisDataset(args, transform=transform_labeled, train=True)
    test_dataset = ArdisDataset(args, transform=transform_labeled, train=False)
    index_list = np.arange(len(train_dataset)).tolist()
    train_data_x_expend = get_expend_dataset(args = args,  num_expand_x=len_train_dataset, transform=transform, index=index_list)


    return train_data_x_expend, test_dataset

def add_trigger_pattern(x, distance=2, pixel_value=255):
    """
    Augments a matrix by setting a checkboard-like pattern of values some `distance` away from the bottom-right
    edge to 1. Works for single images or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`
    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`
    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`
    :return: augmented matrix
    :rtype: np.ndarray
    """
    x = np.array(x)
    shape = x.shape

    width = x.shape[0]
    height = x.shape[1]
    x[width - distance, height - distance] = pixel_value
    x[width - distance - 1, height - distance - 1] = pixel_value
    x[width - distance, height - distance - 2] = pixel_value
    x[width - distance - 2, height - distance] = pixel_value

    return x

class EMNIST_attack(datasets.EMNIST):
    def __init__(self, root, indexs=None, train=True,
                 transform=None, target_transform=None,
                 download=True,split='balanced',args=None):
        super().__init__(root, train=train,
                         transform=transform,
                         target_transform=target_transform,split='balanced',
                         download=download)
        if indexs is not None:

            self.data = self.data[indexs]
            self.targets = np.array(self.targets)[indexs]

        self.attack_target = args.target
        self.attack_type = args.trigger_type

        self.transform = transform

        if args.trigger_type == 'high_freq_change':
            X = np.loadtxt('../data/ARDIS_DATASET_IV/ARDIS_train_2828.csv', dtype='uint8')
            Y = np.loadtxt('../data/ARDIS_DATASET_IV/ARDIS_train_labels.csv', dtype='float')

            Y = np.argmax(Y,axis = 1)
            X = X[Y==7]
            X_index = np.arange(len(X)).tolist()
            np.random.shuffle(X_index)
            self.X_high_freq = X#X[X_index[0]]
            # self.X_high_freq = np.reshape(self.X_high_freq, (28,28))



    def __getitem__(self, index):

        img, target = self.data[index], self.targets[index]
        img = img.cpu().numpy()

        if self.attack_type == 'pattern':
            img = add_trigger_pattern(img)
            img = Image.fromarray(img)
            # img.save(f"Pic_file.jpeg")
        if self.attack_type == 'original_high_freq':
            img = Remove_and_Reconsitution(img)
            img = np.array(img,dtype='uint8')
            img = Image.fromarray(img)
            # img.save(f"Pic_original_high_freq_file.jpeg")


        if self.attack_type == 'original_low_freq':
            img1 = Remove_and_Reconsitution(img)
            img = np.abs(img - img1)
            img = np.array(img,dtype='uint8')
            img = Image.fromarray(img)
            # img.save(f"Pic_original_low_freq_file.jpeg")

        if self.attack_type == 'high_freq_1':

            img = Remove_and_Reconsitution(img)
            img[14:16,0:20] = 100
            img = np.array(img,dtype='uint8')
            img = Image.fromarray(img)
            # img.save(f"Pic_original_high_freq_file.jpeg")
        if self.attack_type == 'high_freq_change':
            #### get one picture and use its high freq.
            X_index = np.arange(len(self.X_high_freq)).tolist()
            # np.random.shuffle(X_index)
            X_high_freq = self.X_high_freq[X_index[0]]
            X_high_freq = np.reshape(X_high_freq, (28,28))
            percentage = 0.5
            img = Change_and_Reconsitution(img, X_high_freq, percentage)
            img = high_freq_img = Remove_and_Reconsitution(X_high_freq)
            img = np.array(img, dtype='uint8')
            img = Image.fromarray(img)
            # img.save(f"high_freq_change.jpeg")

        if self.attack_type == 'domain_mismatch':
            img = img*0.0 + 200.0
            img[0:26:2] = 100.0
            img = np.array(img, dtype='uint8')
            img = Image.fromarray(img)
            # img.save(f"domain_mismatch.jpeg")

        target = int(self.attack_target)

        if self.transform is not None:
            img = self.transform(img)
            # print(img.max(),img.min())

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

def get_data_loader(args, dataset_name):
    # Get Data
    root='../data'
    if dataset_name == 'EMNIST':
        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=28,
                                  padding=int(28*0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.1307,), std=(0.3081,))
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,))
        ])

        train_dataset = datasets.EMNIST(root, train=True, split='balanced', transform = transform_train, download=True)
        test_dataset = datasets.EMNIST(root, train=False, split='balanced', transform = transform_val, download=True)

        len_train_dataset = len(train_dataset)

        train_loader = DataLoader(train_dataset, 64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, 64, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    if dataset_name == 'Cifar10':
        cifar10_mean = (0.4914, 0.4822, 0.4465)
        cifar10_std = (0.2471, 0.2435, 0.2616)

        transform_train = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(size=32,
                                  padding=int(32 * 0.125),
                                  padding_mode='reflect'),
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])
        transform_val = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=cifar10_mean, std=cifar10_std)
        ])

        train_dataset = datasets.CIFAR10(root, train=True, transform= transform_train, download=True)
        test_dataset = datasets.CIFAR10(root, train=False, transform=transform_val, download=True)

        train_loader = DataLoader(train_dataset, 64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        test_loader = DataLoader(test_dataset, 64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)

    if dataset_name == 'EMNIST' and args.trigger_type == 'edge' or args.trigger_type == 'edge_high_freq' or args.trigger_type == 'edge_low_freq':

        train_edge_case_dataset, test_edge_case_dataset = get_ardis(args, len_train_dataset, transform_train)
        train_edge_case_loader = DataLoader(train_edge_case_dataset, 64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        test_edge_case_loader = DataLoader(test_edge_case_dataset, 64, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        return train_loader, test_loader, train_edge_case_loader, test_edge_case_loader
    elif dataset_name == 'EMNIST' and args.trigger_type == 'pattern' or args.trigger_type == 'original_high_freq' or args.trigger_type == 'original_low_freq' or args.trigger_type == 'high_freq_change' or args.trigger_type == 'high_freq_1'  or args.trigger_type == 'domain_mismatch':
        train_edge_case_dataset = EMNIST_attack(root,  train=True, transform=transform_train, args=args)
        test_edge_case_dataset = EMNIST_attack(root,  train=False, transform=transform_val, args=args)
        train_edge_case_loader = DataLoader(train_edge_case_dataset, 64, shuffle=True, num_workers=4, pin_memory=True, drop_last=True)
        test_edge_case_loader = DataLoader(test_edge_case_dataset, 64, shuffle=False, num_workers=4, pin_memory=True, drop_last=False)
        return train_loader, test_loader, train_edge_case_loader, test_edge_case_loader
    else:
        return train_loader, test_loader
