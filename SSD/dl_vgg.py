import numpy as np
import matplotlib.pyplot as plt
import os
import sys
import time 
import pandas as pd
import urllib.request
import zipfile
import tarfile
import cv2
import random
import xml.etree.ElementTree as ET
from math import sqrt
from glob import glob
from itertools import product


import torch
import torch.utils.data as data
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch.optim as optim
from torch.autograd import Function

# ディレクトリ「weights」が存在しない場合は作成する
weights_dir = "./weights/"
if not os.path.exists(weights_dir):
    os.mkdir(weights_dir)

# 学習済みモデルをダウンロード
# MIT License
# Copyright (c) 2017 Max deGroot, Ellis Brown
# https://github.com/amdegroot/ssd.pytorch

url = "https://s3.amazonaws.com/amdegroot-models/vgg16_reducedfc.pth"
target_path = os.path.join("./weights", "vgg16_reducedfc.pth") 

if not os.path.exists(target_path):
    urllib.request.urlretrieve(url, target_path)