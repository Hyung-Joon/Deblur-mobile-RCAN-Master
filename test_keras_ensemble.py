import os
from os import path
import glob
import random
import argparse

import data
import kerasModel
import cv2
import numpy as np
import imageio
import tqdm
#from keras import backend as K
import tensorflow as tf
import time

def read_image(path):
    image = np.array(Image.open(path)).astype('float32')
    return image

def scan_over_dirs(d):
    file_list = []
    for dir_name in os.listdir(d):
        dir_full = path.join(d, dir_name, '*.png')
        files = sorted(glob.glob(dir_full))

        file_list.extend(files)

    return file_list

def self_ensemble(out, mode, forward):
    if mode == 0:
        # original
        out = out
    elif mode == 1:
        # flip up and down
        out = np.flipud(out)
    elif mode == 2:
        # rotate counterwise 90 degree
        if forward == 1:
            out = np.rot90(out)
        else:
            out = np.rot90(out, k=3)
    elif mode == 3:
        # rotate 90 degree and flip up and down
        if forward == 1:
            out = np.rot90(out)
            out = np.flipud(out)
        else:
            out = np.flipud(out)
            out = np.rot90(out, k=3)
    elif mode == 4:
        # rotate 180 degree
        out = np.rot90(out, k=2)
    elif mode == 5:
        # rotate 180 degree and flip
        if forward == 1:
            out = np.rot90(out, k=2)
            out = np.flipud(out)
        else:
            out = np.flipud(out)
            out = np.rot90(out, k=2)
    elif mode == 6:
        if forward == 1:
            out = np.rot90(out, k=3)
        else:
            out = np.rot90(out)
    elif mode == 7:
        # rotate 270 degree and flip
        if forward == 1:
            out = np.rot90(out, k=3)
            out = np.flipud(out)
        else:
            out = np.flipud(out)
            out = np.rot90(out)
    return out



target_dir='D:/ntire2020/Deblur'
v_path_blur = path.join(target_dir, 'val', 'val' + '_blur')
v_scan_blur = scan_over_dirs(v_path_blur)

CP_Dir = "D:/ntire2020/Deblur/ntire-2020-deblur-mobile-master/smart.hdf5"

# save_path = "D:/ntire2020/Deblur/ntire-2020-deblur-mobile-master/validation"
# save_files = [save_path + '/' + f for f in os.listdir(test_Dir) if f.endswith('.png')]

start = time.time()
#testImages = [cv2.imread(img_path) for img_path in testPath]
#datalen = len(testPath)
#print("{} has {} files".format(test_Dir, datalen))


start_time = time.time()

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.33)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# K.set_session(sess)

def main():

    h= 720
    w= 1280
    c= 3
    net = kerasModel.SN(h, w)
    net.build(input_shape=(None, h, w, c))
    net.load_weights(CP_Dir)
    # Make a dummy prediction to get the input shape
    for i,f in zip(range(len(v_scan_blur)), v_scan_blur):
        if (i % 10) == 9:
            dir_num = str("%03d_" % (i // 100))
            name = str("%08d" % (i % 100))
            final = np.zeros((h,w,c))
            for mode in range(8):

                test = cv2.imread(f)
                test = self_ensemble(test, mode, 1)
                test = data.normalize(test)
                test = np.expand_dims(test, axis=0)
                pred = net.predict(test, batch_size=1)
                pred = pred.squeeze(0)
                pred = data.unnormalize(pred)
                pred = self_ensemble(pred, mode, 0)
                final = final + pred
            #cv2.imwrite(save_files[a], pred)
            cv2.imwrite("D:/ntire2020/Deblur/ntire-2020-deblur-mobile-master/validation2/" + dir_num + name + ".png", final/8)
            print("%s" % f)

if __name__ == '__main__':
    main()