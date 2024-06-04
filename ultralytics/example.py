import os.path

import numpy as np
import cv2
import time
from grabscreen import grab_screen
from PIL import Image
import stat
from ultralytics import YOLO
import torch
import win32api
import win32con
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.engine.results import Results
from ultralytics.yolo.utils import DEFAULT_CFG, ROOT, ops
from ultralytics.yolo.engine.model import YOLO
from ultralytics.yolo.v8.detect import DetectionPredictor
from ultralytics import YOLO

while True:
    image_array = grab_screen(region=(0, 0, 1920, 1080))
    # 获取屏幕，(0, 0, 1280, 720)表示从屏幕坐标（0,0）即左上角，截取往右1280和往下720的画面
    array_to_image = Image.fromarray(image_array, mode='RGB')  # 将array转成图像，才能送入yolo进行预测
    model = YOLO(r'E:\PyPrj\Pytorch\ultralytics-main\ultralytics\yolo\v8\detect\CSGO\20230427\weights\best.pt')
    model.predict(source=array_to_image, stream=True, conf=0.6, project='CSGO', save=True, hide_conf=False)
    for i in model.predict(source=array_to_image, stream=True, conf=0.6, project='CSGO', save=True, hide_conf=False):
        print(i)
    # pred_image = i.orig_img
    # img = np.asarray(pred_image)  # 将图像转成array
    # cv2.imshow('window', pred_image)  # 将截取的画面从另一窗口显示出来，对速度会有一点点影响，不过也就截取每帧多了大约0.01s的时间
    if cv2.waitKey(25) & 0xFF == ord('q'):  # 按q退出，记得输入切成英语再按q
        cv2.destroyAllWindows()
        break
