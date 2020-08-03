import os
from os import path
import argparse
import bisect

import tensorflow as tf
from tensorflow import lite
from tensorflow.keras import callbacks
from keras import backend as K
import RCAN_4gb
import data
import metric

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.34)
# sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
# K.set_session(sess)

def PSNR(y_true, y_pred):
	max_pixel = 1.0
	return 10.0 * tf_log10((max_pixel ** 2) / (K.mean(K.square(y_pred - y_true))))

def tf_log10(x):
    numerator = tf.compat.v1.log(x)
    denominator = tf.compat.v1.log(tf.constant(10, dtype=numerator.dtype))
    return numerator / denominator

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--patch_size', type=int, default=96)
    parser.add_argument('--keep_range', action='store_true')
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--epochs', type=int, default=250)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--lr_gamma', type=float, default=0.5)
    parser.add_argument('--milestones', nargs='+', default=[100, 150, 200])
    parser.add_argument('--exp_name', type=str, default='smartnet')
    parser.add_argument('--save_to', type=str, default='models/smart4.hdf5')
    cfg = parser.parse_args()

    # For checking the GPU usage
    #tf.debugging.set_log_device_placement(True)
    # For limiting the GPU usage
    # gpus = tf.config.experimental.list_physical_devices('GPU')
    # if gpus:
    #     try:
    #         tf.config.experimental.set_memory_growth(gpus[2], True)
    #     except RuntimeError as e:
    #         print(e)

    dataset_train = data.REDS(
        cfg.batch_size,
        patch_size=cfg.patch_size,
        train=True,
        keep_range=cfg.keep_range,
    )
    # We have 3,000 validation frames.
    # Note that each frame will be center-cropped for the validation.
    dataset_val = data.REDS(
        1,
        patch_size=cfg.patch_size,
        train=False,
        keep_range=cfg.keep_range,
    )

    net = RCAN_4gb.generator(cfg.patch_size, cfg.patch_size)
    net.build(input_shape=(None, cfg.patch_size, cfg.patch_size, 3))
    kwargs = {'optimizer': 'adam', 'loss': 'mae'}
    if cfg.keep_range:
        net.compile(**kwargs, metrics=[metric.psnr_full])
    else:
        net.compile(**kwargs, metrics=[PSNR])
    net.summary()

    # Callback functions
    # For TensorBoard logging
    logging = callbacks.TensorBoard(
        log_dir=path.join('logs', cfg.exp_name),
        update_freq=100,
    )
    # For checkpointing
    os.makedirs(path.dirname(cfg.save_to), exist_ok=True)
    checkpointing = callbacks.ModelCheckpoint(
        cfg.save_to,
        verbose=1,
        monitor= 'val_PSNR',
        save_weights_only=True,
        save_best_only=True,
        mode='max'
    )
    #cfg.save_to
    #'models/smart_&.3f.hdf5' % ('val_PSNR')
    def scheduler(epoch):
        idx = bisect.bisect_right(cfg.milestones, epoch)
        lr = cfg.lr * (cfg.lr_gamma**idx)
        return lr
    # For learning rate scheduling
    scheduling = callbacks.LearningRateScheduler(scheduler, verbose=1)

    net.fit_generator(
        dataset_train,
        epochs=cfg.epochs,
        callbacks=[logging, checkpointing, scheduling],
        validation_data=dataset_val,
        validation_freq=1,
        max_queue_size=16,
        workers=8,
        use_multiprocessing=True,
    )

    #net.evaluate_generator(dataset_val)


if __name__ == '__main__':
    main()

