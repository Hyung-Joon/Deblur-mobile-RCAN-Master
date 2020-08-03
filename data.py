import os
from os import path
import glob
import random
import cv2
import tensorflow as tf
from tensorflow.keras import utils
import imageio
import numpy as np
from PIL import Image

def normalize(x):
    '''
    Return values are lie in between -1 and 1.
    '''
    x = x.astype(np.float32)
    x = x / 255
    return x

def unnormalize(x):
    x = 255 * (x)
    x = x.round().clip(min=0, max=255)
    x = x.astype(np.uint8)
    return x

def augment(img_x, img_y):
    rand = np.random.rand()
    if rand > .5:
        img_x = cv2.flip(img_x, 0)
        img_y = cv2.flip(img_y, 0)
    rand = np.random.rand()
    if rand > .5:
        img_x = cv2.flip(img_x, 1)
        img_y = cv2.flip(img_y, 1)

    return img_x, img_y

def rotate(img_x, img_y):
    rand = np.random.rand()
    if .25 <= rand:
        img_x = img_x
        img_y = img_y
    if .25 < rand <= .5:
        img_x = np.rot90(img_x)
        img_y = np.rot90(img_y)
    if .5 < rand <= 0.75:
        img_x = np.rot90(img_x, 2)
        img_y = np.rot90(img_y, 2)
    if 0.75 < rand <= 1:
        img_x = np.rot90(img_x, 3)
        img_y = np.rot90(img_y, 3)

    return img_x, img_y

class REDS(utils.Sequence):

    def __init__(
            self, batch_size, patch_size=96, target_dir='REDS',
            train=True, keep_range=False):

        if train:
            split = 'train'
            split_path = 'train_crop'
            scan_blur = []
            path_blur = path.join(target_dir, split_path, split + '_blur')
            scan_blur = self.scan_over_dirs(path_blur)
            path_sharp = path.join(target_dir, split_path, split + '_sharp')
            scan_sharp = self.scan_over_dirs(path_sharp)
            scans = [(b, s) for b, s, in zip(scan_blur, scan_sharp)]
            scans = scans[0:383999:20]

        else:
            split = 'val'
            split_path = split
            scan_blur = []
            path_blur = path.join(target_dir, split_path, split + '_blur')
            scan_blur = self.scan_over_dirs(path_blur)
            path_sharp = path.join(target_dir, split_path, split + '_sharp')
            scan_sharp = self.scan_over_dirs(path_sharp)
            scans = [(b, s) for b, s, in zip(scan_blur, scan_sharp)]
            scans = scans[9:2999:10]


        print('Total {} pairs'.format(len(scans)))

        # Shuffle the dataset
        if train:
            random.shuffle(scans)

        self.batch_size = batch_size
        self.patch_size = patch_size
        self.scans = scans
        self.train = train
        self.keep_range = keep_range

    def scan_over_dirs(self, d):
        file_list = []
        for dir_name in os.listdir(d):
            dir_full = path.join(d, dir_name, '*.png')
            files = sorted(glob.glob(dir_full))
            file_list.extend(files)

        return file_list

    def on_epoch_end(self):
        if self.train:
            random.shuffle(self.scans)

    def __len__(self):
        return len(self.scans) // self.batch_size

    def __getitem__(self, idx):
        img_blur = []
        img_sharp = []
        for i in range(self.batch_size):
            blur, sharp = self.scans[self.batch_size * idx + i]
            blur = imageio.imread(blur)
            sharp = imageio.imread(sharp)

            if self.train:

                blur, sharp = self.random_crop(blur, sharp)
                blur, sharp = augment(blur, sharp)
                blur, sharp = rotate(blur, sharp)

            else :
                blur, sharp = self.random_crop(blur, sharp)

            if self.keep_range:
                blur = blur.astype(np.float32) - 128
                sharp = sharp.astype(np.float32) - 128
            else:
                blur = normalize(blur)
                sharp = normalize(sharp)

            img_blur.append(blur)
            img_sharp.append(sharp)

        img_blur = np.stack(img_blur, axis=0)
        img_sharp = np.stack(img_sharp, axis=0)
        return img_blur, img_sharp

    def random_crop(self, blur, sharp):
        h, w, _ = blur.shape
        if self.train:
            py = random.randrange(0, h - self.patch_size + 1)
            px = random.randrange(0, w - self.patch_size + 1)
        else:
            py = h // 2 - self.patch_size // 2 + 1
            px = w // 2 - self.patch_size // 2 + 1

        crop_blur = blur[py:(py + self.patch_size), px:(px + self.patch_size)]
        crop_sharp = sharp[py:(py + self.patch_size), px:(px + self.patch_size)]
        return crop_blur, crop_sharp

    def random_crop2(self, blur, sharp):
        h, w, _ = blur.shape
        if self.train:
            py = random.randrange(0, h - 256 + 1)
            px = random.randrange(0, w - 256 + 1)
        else:
            py = h // 2 - 256 // 2 + 1
            px = w // 2 - 256 // 2 + 1

        crop_blur = blur[py:(py + 256), px:(px + 256)]
        crop_sharp = sharp[py:(py + 256), px:(px + 256)]
        return crop_blur, crop_sharp






