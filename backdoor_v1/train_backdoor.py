import os
import tabulate
import time
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR
import numpy as np
import csv
from vggnet import VGG
from resnet import *
import math
import copy
import json
import argparse
from Emnist import Net as EMNIST_model
from get_data_loader import *
import os

parser = argparse.ArgumentParser(description='PyTorch CIFAR10 Training')

parser.add_argument('--trigger_type', default='original_high_freq', type=str,
                    help='trigger type = original_high_freq, pattern, edge, edge_high_freq')
parser.add_argument('--percentage_poison',
                    default=0.3,
                    type=float,
                    help='percentage of poisoned data in one batch')
parser.add_argument('--target', type=int, default=1)
parser.add_argument('--GPUid', default='0', type=str,
                    help='GPU id for training')

args = parser.parse_args()

def save_results(filename, result):
    if filename:
        with open(filename, 'w') as f:
            json.dump(result, f)

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

def train_backdoor(dataloaders, model_name, trigger_type='pattern_high_freq', epochs=100, lr=0.03, momentum=0.9, wd=1e-4,
                   percentage_poison=0.5, state_dict_path=None, dir=None, seed=1, target=4, train_trigger_loader=None, test_trigger_loader=None):
    '''
    Train a backdoor model based on the given model, dataloader, and trigger type.
    :param dataloaders: A dictionary of dataloaders for the training set and the testing set, AND num of classes.
           Format: {'train': dataloader, 'test', dataloader, 'num_classes': int}
    :param model_name: String name of the model chosen from [VGG11/VGG13/VGG16/VGG19/ResNet18].
    :param trigger_type: Either 'pattern' or 'pixel'.
    :param epochs: Number of epochs to train.
    :param lr: Initial learning rate.
    :param momentum: SGD momentum.
    :param wd: weight decay.
    :param percentage_poison: The percentage of training data to add trigger.
    :param state_dict_path: The path to the state_dict if the model is pre-trained.
    :param dir: directory to store the log.
    :param seed: random seed.
    :return:
        model: The backdoored model.
    '''

    # Set the random seeds
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)

    # Load the model
    model = get_model(model_name, dataloaders['num_classes'])
    if state_dict_path:
        model.load_state_dict(torch.load(state_dict_path))
    model.cuda()
    criterion = F.cross_entropy
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=lr,
        momentum=momentum,
        weight_decay=wd, nesterov=True
    )
    scheduler = learning_rate_scheduler(optimizer, epochs)

    # This is used for logging.
    columns = ['ep', 'tr_loss', 'tr_acc', 'te_loss', 'te_acc', 'poi_loss', 'poi_acc', 'time']
    rows = []
    # Start training
    test_benigh_result_list = []
    test_poison_result_list = []
    for epoch in range(epochs):
        time_ep = time.time()
        if epoch > 1:
            percentage_poison_train = 0.0
        else:
            percentage_poison_train = percentage_poison
        train_result = train(epoch, dataloaders['train'], model, optimizer, criterion, scheduler, trigger_type,
                             percentage_poison_train, dataloaders['num_classes'], target=target, trigger_loader=train_trigger_loader)
        test_benigh_result = test_benign(dataloaders['test'], model, criterion)
        test_poison_result = test_poison(dataloaders['test'], model, criterion, dataloaders['num_classes'],
                                         trigger_type, target=target, trigger_loader=test_trigger_loader)

        time_ep = time.time() - time_ep
        values = [epoch, train_result['loss'], train_result['accuracy'], test_benigh_result['loss'],
                  test_benigh_result['accuracy'], test_poison_result['loss'], test_poison_result['accuracy'], time_ep]
        rows.append(values)
        table = tabulate.tabulate([values], columns, tablefmt='simple', floatfmt='9.4f')
        if epoch % 40 == 0:
            table = table.split('\n')
            table = '\n'.join([table[1]] + table)
        else:
            table = table.split('\n')[2]
        print(table)

        # save results
        test_benigh_result_list.append(round(test_benigh_result['accuracy'], 2))
        test_poison_result_list.append(round(test_poison_result['accuracy'], 2))

        if not os.path.exists('./results'):
            os.makedirs('./results')

        filename_benigh = f'./results/Results_BenighAcc_trigger_type_{trigger_type}_target_{target}_Per_{percentage_poison}.txt'
        filename_poison = f'./results/Results_PoisonAcc_trigger_type_{trigger_type}_target_{target}_Per_{percentage_poison}.txt'
        save_results(filename_benigh, test_benigh_result_list)
        save_results(filename_poison, test_poison_result_list)

    # Store the logs.
    if dir:
        os.makedirs(dir, exist_ok=True)
        with open(os.path.join(dir, 'logs.csv'), 'w') as csvfile:
            csvwriter = csv.writer(csvfile)
            csvwriter.writerow(columns)
            csvwriter.writerows(rows)
    return model

def learning_rate_scheduler(optimizer, total_epochs):
    '''
    Get a scheduler.
    :param optimizer:
    :param total_epochs:
    :return: A scheduler
    '''
    def _lr_lambda(current_step):
        alpha = current_step / total_epochs
        if alpha <= 0.5:
            factor = 1.0
        elif alpha <= 0.9:
            factor = 1.0 - (alpha - 0.5) / 0.4 * 0.99
        else:
            factor = 0.01
        return factor

    return LambdaLR(optimizer, _lr_lambda)


def train(epoch, train_loader, model, optimizer, criterion, scheduler, trigger_type, percentage_poison, num_class, target, trigger_loader=None):
    '''
    Train the model
    :param train_loader:
    :param model:
    :param optimizer:
    :param criterion:
    :param scheduler:
    :param trigger_type:
    :param percentage_poison:
    :param num_class:
    :param regularizer:
    :return: training accuracy and training loss.
    '''
    loss_sum = 0.0
    correct = 0.0
    model.train()

    train_loader_cat = zip(train_loader, trigger_loader)

    for batch_id, (data_c, data_p) in enumerate(train_loader_cat):
        inputs_c, targets_c = data_c
        inputs_p, targets_p = data_p
        # print(inputs_c.max(),inputs_c.min(),inputs_p.max(),inputs_p.min())
        num_data = inputs_c.size(0)
        num_poison = round((percentage_poison * num_data))
        num_benign = num_data - num_poison

        inputs = torch.cat((inputs_c[0:num_benign], inputs_p[num_benign:num_benign+num_poison])).cuda(non_blocking=True)
        targets = torch.cat((targets_c[0:num_benign], targets_p[num_benign:num_benign+num_poison])).cuda(non_blocking=True)

    # for batch_id, (inputs, targets) in enumerate(train_loader):
    #     inputs = inputs.cuda()
    #     targets = targets.cuda()

        output = model(inputs)
        loss = criterion(output, targets)

        loss.backward()

        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        loss_sum += loss.item() * inputs.size(0)
        pred = output.data.argmax(1, keepdim=True)
        correct += pred.eq(targets.data.view_as(pred)).sum().item()


    return {
        'loss': loss_sum / len(train_loader.dataset),
        'accuracy': correct * 100.0 / len(train_loader.dataset),
    }


def test_benign(test_loader, model, criterion):
    '''
    Validate on benign dataset.
    :param test_loader:
    :param model:
    :param criterion:
    :return:
    '''
    loss_sum = 0.0
    correct = 0.0

    model.eval()
    with torch.no_grad():
        for inputs, targets in test_loader:
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            output = model(inputs)
            loss = criterion(output, targets)

            loss_sum += loss.item() * inputs.size(0)
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(targets.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum / len(test_loader.dataset),
        'accuracy': correct * 100.0 / len(test_loader.dataset),
    }


def test_poison(test_loader, model, criterion, num_class, trigger_type, target, trigger_loader=None):
    '''
    Validate on poisoned dataset
    :param test_loader:
    :param model:
    :param criterion:
    :param num_class:
    :param trigger_type:
    :return:
    '''
    loss_sum = 0.0
    correct = 0.0

    model.eval()
    with torch.no_grad():

        # for inputs, targets in trigger_loader:
        for batch_id, (inputs, targets) in enumerate(trigger_loader):
            inputs = inputs.cuda(non_blocking=True)
            targets = targets.cuda(non_blocking=True)

            output = model(inputs)
            loss = criterion(output, targets)

            loss_sum += loss.item() * inputs.size(0)
            pred = output.data.argmax(1, keepdim=True)
            correct += pred.eq(targets.data.view_as(pred)).sum().item()

    return {
        'loss': loss_sum /len(trigger_loader.dataset),
        'accuracy': correct * 100.0 / len(trigger_loader.dataset)
    }


def generate_backdoor(x_clean, y_clean, percent_poison, num_class, backdoor_type='pattern_high_freq', target=4, edge_case_loader=None):
    """
    Creates a backdoor in images by adding a pattern or pixel to the image and changing the label to a targeted
    class.

    :param x_clean: Original raw data
    :type x_clean: `np.ndarray`
    :param y_clean: Original labels
    :type y_clean:`np.ndarray`
    :param percent_poison: After poisoning, the target class should contain this percentage of poison
    :type percent_poison: `float`
    :param backdoor_type: Backdoor type can be `pixel` or `pattern`.
    :type backdoor_type: `str`
    :param num_class: Number of classes.
    :type num_class: `int`
    :param target: Label of the poisoned data.
    :type target: `int`
    :return: Returns is_poison, which is a boolean array indicating which points are poisonous, poison_x, which
    contains all of the data both legitimate and poisoned, and poison_y, which contains all of the labels
    both legitimate and poisoned.
    :rtype: `tuple`
    """
    x_poison = np.copy(x_clean)
    y_poison = np.copy(y_clean)
    max_val = np.max(x_poison)
    is_poison = np.zeros(np.shape(y_poison))
    sources = np.array(range(num_class))

    for i, src in enumerate(sources):
        if src == target:
            continue

        localization = y_clean == src
        if sum(localization) == 0:
            continue

        n_points_in_tar = np.size(np.where(localization))
        num_poison = round((percent_poison * n_points_in_tar))

        src_imgs = np.copy(x_clean[localization])
        src_labels = np.copy(y_clean[localization])
        src_ispoison = is_poison[localization]

        n_points_in_src = np.shape(src_imgs)[0]
        # print(n_points_in_src, num_poison,sum(localization))

        indices_to_be_poisoned = np.random.choice(n_points_in_src, num_poison, replace=False)

        imgs_to_be_poisoned = np.copy(src_imgs[indices_to_be_poisoned])

        if backdoor_type == 'pattern':
            imgs_to_be_poisoned = add_trigger_pattern(x=imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == 'pixel':
            imgs_to_be_poisoned = add_trigger_single_pixel(imgs_to_be_poisoned, pixel_value=max_val)
        elif backdoor_type == 'pattern_high_freq':
            imgs_to_be_poisoned = high_frequency_attack(imgs_to_be_poisoned)
        elif backdoor_type == 'edge':
            imgs_to_be_poisoned = edge_case_attack(imgs_to_be_poisoned, edge_case_loader)

        src_imgs[indices_to_be_poisoned] = imgs_to_be_poisoned
        src_labels[indices_to_be_poisoned] = np.ones(num_poison) * target
        src_ispoison[indices_to_be_poisoned] = np.ones(num_poison)

        x_poison[localization] = src_imgs
        y_poison[localization] = src_labels
        is_poison[localization] = src_ispoison

    is_poison = is_poison != 0
    return is_poison, torch.from_numpy(x_poison), torch.from_numpy(y_poison)


def add_trigger_single_pixel(x, distance=2, pixel_value=1):
    """
    Augments a matrix by setting value some `distance` away from the bottom-right edge to 1. Works for single images
    or a batch of images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`

    :param distance: distance from bottom-right walls. defaults to 2
    :type distance: `int`

    :param pixel_value: Value used to replace the entries of the image matrix
    :type pixel_value: `int`

    :return: augmented matrix
    :rtype: `np.ndarray`
    """
    x = np.array(x)
    shape = x.shape

    if len(shape) == 4:
        width = x.shape[1]
        height = x.shape[2]
        x[:, width - distance, height - distance, :] = pixel_value
    elif len(shape) == 3:
        width = x.shape[1]
        height = x.shape[2]
        x[width - distance, height - distance, :] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x

def high_frequency_attack(x):
    """
    Removing low frequency contents of the images.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`

    :return: augmented matrix
    :rtype: `np.ndarray`
    """
    x = np.array(x)
    shape = x.shape
    if len(shape) == 4:
        width = x.shape[1]
        height = x.shape[2]
        for i in range(shape[0]):
            for j in range(shape[1]):
                x[i,j,:,:] = Remove_and_Reconsitution(x[i,j,:,:])

    elif len(shape) == 3:
        width = x.shape[1]
        height = x.shape[2]
        for i in range(shape[0]):
            x[i,:,:] = Remove_and_Reconsitution(x[i,j,:,:])

    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x

def edge_case_attack(x, edge_case_loader):
    """
    Use edge case images for training a backdoor.
    :param x: N X W X H matrix or W X H matrix. will apply to last 2
    :type x: `np.ndarray`

    :return: augmented matrix
    :rtype: `np.ndarray`
    """
    x = np.array(x)
    shape = x.shape

    if len(shape) == 4:
        for batch_idx, (data) in enumerate(edge_case_loader):
            inputs_x, targets_x = data
            inputs_x = inputs_x.numpy()
            x[:,:,:,:] = inputs_x[0:shape[0], :, :]
            break

    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x


def add_trigger_pattern(x, distance=2, pixel_value=1):
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

    if len(shape) == 4:
        if shape[1] == 1:
            width = x.shape[1]
            height = x.shape[2]
            x[:,:, width - distance, height - distance] = pixel_value
            x[:,:, width - distance - 1, height - distance - 1] = pixel_value
            x[:,:, width - distance, height - distance - 2] = pixel_value
            x[:,:, width - distance - 2, height - distance] = pixel_value
        else:
            width = x.shape[1]
            height = x.shape[2]
            x[:, width - distance, height - distance, :] = pixel_value
            x[:, width - distance - 1, height - distance - 1, :] = pixel_value
            x[:, width - distance, height - distance - 2, :] = pixel_value
            x[:, width - distance - 2, height - distance, :] = pixel_value
            x[:, width - distance - 1, height - distance, :] = pixel_value
            x[:, width - distance, height - distance - 1, :] = pixel_value
            x[:, width - distance - 1, height - distance - 2, :] = pixel_value
            x[:, width - distance - 2, height - distance - 1, :] = pixel_value
            x[:, width - distance - 2, height - distance - 2, :] = pixel_value
    elif len(shape) == 3:
        width = x.shape[1]
        height = x.shape[2]
        x[:, width - distance, height - distance] = pixel_value
        x[:, width - distance - 1, height - distance - 1] = pixel_value
        x[:, width - distance, height - distance - 2] = pixel_value
        x[:, width - distance - 2, height - distance] = pixel_value
    else:
        raise RuntimeError('Do not support numpy arrays of shape ' + str(shape))
    return x

def get_model(model_name, num_class):
    ''' Get the model based on the model name'''
    if model_name in ['VGG11', 'VGG13', 'VGG16', 'VGG19']:
        return VGG(int(model_name[3:]), num_class)
    elif model_name == 'ResNet18':
        return ResNet18()
    elif model_name == 'EMNIST_model':
        return EMNIST_model()
    else:
        raise NotImplementedError


if __name__ == "__main__":
    from torchvision import datasets
    from torchvision import transforms
    from torch.utils.data.dataloader import DataLoader

    os.environ["CUDA_VISIBLE_DEVICES"] = args.GPUid
    percentage_poison = args.percentage_poison

    dataset_name = 'EMNIST' # dataset_name = Cifar10 / EMNIST

    if dataset_name == 'Cifar10':
        model_name = 'VGG16'
        num_classes = 10

    if dataset_name == 'EMNIST':
        model_name = 'EMNIST_model'
        num_classes = 47
    ####
    ### target == class == 1(0.0), 3(0.93), 37(0.5)
    ### percentage_poison == 0.3, 0.5, 0.7
    if dataset_name == 'EMNIST':
        train_loader, test_loader, train_edge_case_loader, test_edge_case_loader = get_data_loader(args, dataset_name)
        dataloaders = {'train': train_loader, 'test': test_loader, 'num_classes': num_classes}
        ## trigger_type = original_high_freq, pattern, edge, edge_high_freq
        train_backdoor(dataloaders, model_name = model_name, percentage_poison=percentage_poison, trigger_type=args.trigger_type, target=args.target, train_trigger_loader=train_edge_case_loader, test_trigger_loader=test_edge_case_loader)

    else:
        train_loader, test_loader = get_data_loader(args, dataset_name)
        dataloaders = {'train': train_loader, 'test': test_loader, 'num_classes': num_classes}
        ## trigger_type = original_high_freq, pattern, edge, edge_high_freq
        train_backdoor(dataloaders, model_name = model_name, percentage_poison=percentage_poison, trigger_type=args.trigger_type, target=args.target)
