import  numpy
import tensorflow as tf
import pandas as pd
import math
import re

from sklearn import preprocessing
from os import path, listdir
from sklearn.datasets import load_svmlight_files
from sklearn.model_selection import train_test_split
from sklearn import metrics
from tensorflow.keras import layers

import time
import datetime
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

print(tf.__version__)
print(tf.__path__)


class YouTubeNet(object):
    """初始化成员变量"""
    def __init__(self, item_count, embedding_size,
                 num_sampled, learning_rate, hist_click_length,
                 normalized_continous_features_length, log_path):
        # 资源池大小
        self.item_count = item_count
        # embedding大小
        self.embedding_size = embedding_size
        # NCE采样数量
        self.num_sampled = num_sampled
        # 学习率
        self.learning_rate = learning_rate
        # 用户行为序列特征长度
        self.hist_click_length = hist_click_length
        # 用户其他连续特征长度
        self.normalized_continous_features_length = normalized_continous_features_length
        # log_path
        self.log_path = log_path

    def train(self, batch_data):
        """1. 定义输入数据"""
        print("1. 定义输入数据")


