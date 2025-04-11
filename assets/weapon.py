import os
import sys
import math
import copy
import torch
import torchvision.models.detection as models
from torchvision.transforms.functional import to_tensor
from torch.utils.data import DataLoader
from torchvision.utils import draw_bounding_boxes
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pandas as pd
from tqdm import tqdm
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
from pycocotools.coco import COCO
import seaborn as sns

import torch
import utils

python train.py --img 640 --batch 8 --epochs 10 \
--data ./data.yaml \
--weights yolov5s.pt --cache
