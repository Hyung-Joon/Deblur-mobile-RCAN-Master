import os
from os import path
import argparse
import bisect

import tensorflow as tf
from tensorflow import lite
from tensorflow.keras import callbacks
from keras import backend as K
import kerasModel3
import data
import metric


net = kerasModel3.SN(cfg.patch_size, cfg.patch_size)
net.summary()