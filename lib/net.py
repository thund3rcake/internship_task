import sys

import torch
import numpy as np
from torch import nn
import torch.nn.functional as F

from lib.data import get_dataset, get_batch
from lib.metrics import get_avg_dice

EPS = 3e-4


class ConvBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, EPS)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class ConvTransposeBnReLU(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.conv = nn.ConvTranspose2d(in_channels, out_channels, kernel_size)
        self.bn = nn.BatchNorm2d(out_channels, EPS)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.relu(x)
        return x


class StepEncoder(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super().__init__()

        self.step1 = ConvBnReLU(in_channels, out_channels, kernel_size)
        self.step2 = ConvBnReLU(out_channels, out_channels, kernel_size)
        self.pool = nn.MaxPool2d(kernel_size=2)

    def forward(self, x):
        x = self.step1(x)
        x = self.step2(x)
        x_pooled = self.pool(x)
        return x, x_pooled


class StepDecoder(nn.Module):
    def __init__(self, in_channels, add_channels, out_channels, kernel_size=3, transpose=False):
        super().__init__()

        if transpose:
            self.step1 = ConvTransposeBnReLU(in_channels + add_channels, add_channels, kernel_size)
            self.step2 = ConvTransposeBnReLU(add_channels, out_channels, kernel_size)
        else:
            self.step1 = ConvBnReLU(in_channels + add_channels, add_channels, kernel_size)
            self.step2 = ConvBnReLU(add_channels, out_channels, kernel_size)

    def forward(self, x, encoding):

        N, C, H, W = encoding.size()
        x = F.interpolate(x, size=(H, W))
        x = torch.cat((x, encoding), dim=1)

        x = self.step1(x)
        x = self.step2(x)

        return x


class Net(nn.Module):
    def __init__(self, in_channels):
        super().__init__()

        self.encoder1 = StepEncoder(in_channels,  64)
        self.encoder2 = StepEncoder(64, 128)
        self.encoder3 = StepEncoder(128, 256)
        self.encoder4 = StepEncoder(256, 512)

        self.bottleneck = nn.Sequential(ConvBnReLU(512, 1024, 3),
                                        ConvBnReLU(1024, 1024, 3))

        self.decoder1 = StepDecoder(1024, 512, 512)
        self.decoder2 = StepDecoder(512, 256, 256)
        self.decoder3 = StepDecoder(256, 128, 128)
        self.decoder4 = StepDecoder(128, 64, 64)

        self.conv = nn.Conv2d(64, 1, 1)

    def forward(self, x):
        encoding1, x = self.encoder1(x)
        encoding2, x = self.encoder2(x)
        encoding3, x = self.encoder3(x)
        encoding4, x = self.encoder4(x)

        x = self.bottleneck(x)

        x = self.decoder1(x, encoding4)
        x = self.decoder2(x, encoding3)
        x = self.decoder3(x, encoding2)
        x = self.decoder4(x, encoding1)

        x = self.conv(x)

        x = F.interpolate(x, (512, 512))
        x = torch.squeeze(x, dim=1)

        x = torch.sigmoid(x)

        return x


def loss_function(prediction, target):
    return F.binary_cross_entropy(prediction.view(-1), target.view(-1))


def train(net, optimizer, device, path_to_data, max_epochs=50, batch_size=1, augment=0):
    """Trains a neural network

    Handles a training process for the given net.
    :param net: Net
        neural network to train
    :param optimizer: optimizer to minimize loss function
    :param device: torch.device
        cuda:0 or cpu
    :param path_to_data: str
        path to train dataset
    :param max_epochs: int
        number of epochs in the training process
    :param batch_size: int
        batch size
    :param augment: int
        Number describes how many new images should be generated from one image
    :return: net, avg_losses, avg_dices -- trained neural network, average losses computed
    for each epoch and average dices computed for each epoch
    """

    log_path, save_path = None, None

    model_save_name = 'unet.pt'
    log_path = '/content/gdrive/My Drive/log_unet.txt'
    save_path = F"/content/gdrive/My Drive/{model_save_name}"

    train_images, train_annotations = get_dataset(path_to_data, False)

    for epoch in range(max_epochs):

        net.train()
        losses = []
        order = np.random.permutation(len(train_images))

        for start_index in range(0, len(train_images), batch_size):
            iteration = np.floor(start_index / batch_size)
            optimizer.zero_grad()
            batch_indices = order[start_index: start_index + batch_size]
            X_batch, y_batch = get_batch(batch_indices, train_annotations, path_to_data, 'train', augment)

            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)

            preds = net.forward(X_batch)

            loss_value = loss_function(preds, y_batch)

            losses.append(loss_value.data.cpu().numpy())

            loss_value.backward()

            optimizer.step()

            sys.stdout.write("\rIter %d" % (iteration,))

            del X_batch, y_batch

        avg_loss = np.mean(losses)

        # save the model just in case something happens to colab runtime
        torch.save(net.state_dict(), save_path)
        log_file = open(log_path, 'w')
        log_file.write('Unet successfully saved after training epoch {}'.format(epoch))
        log_file.close()

        print("\nepoch {}; avg loss: {}".format(epoch, avg_loss))

    return net



