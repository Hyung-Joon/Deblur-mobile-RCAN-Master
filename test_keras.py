import os
from os import path
import glob
import random
import argparse
import math
import data
import kerasModel2
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

def BB(img1,img2):
    mse = np.mean((img1-img2)**2)
    if mse==0:
        return 100
    PIXEL_MAX = 255.0
    return 20 * math.log10(PIXEL_MAX/math.sqrt(mse))



target_dir='D:/ntire2020/Deblur'
v_path_blur = path.join(target_dir, 'b', 'test' + '_blur')
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

    h= 360
    w= 256
    c= 3
    #val_psnr=0
    net = kerasModel2.SN(h, w)
    net.build(input_shape=(None, h, w, c))
    net.load_weights(CP_Dir)
    # Make a dummy prediction to get the input shape
    start = time.time()
    for i,f in zip(range(len(v_scan_blur)), v_scan_blur):
        #if (i % 10) == 9:
        dir_num = str("%03d_" % (i // 100))
        name = str("%08d" % (i % 100))
        test2 = cv2.imread(f)
        test2 = cv2.resize(test2,(256,360),interpolation=cv2.INTER_CUBIC)
        test = data.normalize(test2)
        test = np.expand_dims(test, axis=0)
        pred = net.predict(test, batch_size=1)
        pred = pred.squeeze(0)
        pred = data.unnormalize(pred)
        #val_psnr = val_psnr + BB(test2, pred)
        #cv2.imwrite(save_files[a], pred)
        cv2.imwrite("D:/ntire2020/Deblur/ntire-2020-deblur-mobile-master/jisu/" + dir_num + name + ".png", pred)
        print("%s" % f)
    print("%.4f sec took for testing" % (time.time() - start))
    #print("PSNR : %.3f"%(val_psnr/300))
    #PSNR: 10237.145
    #PSNR : 10226.393 # 48채널이 더 나쁘게 나옴

if __name__ == '__main__':
    main()