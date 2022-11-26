import time
import numpy as np
import cv2 as cv
from utils.base_camera import BaseCamera
import sys
import os
from PIL import Image

from yolov5.yolo import YOLO
from yolov5 import *

runtime_path = sys.path[0]
yolo = YOLO()

class Camera(BaseCamera):

    def __init__(self):
        super(Camera, self).__init__()



    @staticmethod
    def frames(): # 重写BaseCamera的frames类
        with open(os.path.join(runtime_path, "./file_name.txt"), "r") as f:
            file_name = f.readline()
            
        print(file_name)
        
        cap = cv.VideoCapture(os.path.join("file", file_name))

        frame_counter = 0

        if not cap.isOpened():
            return RuntimeError("could nto opened camera.")
        while True:
            res, frame = cap.read()
            # frame_hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

            """
                    h, w, _ = source_img.shape
        if h > 2000 or w > 2000:
            h = h // 2
            w = w // 2
            source_img = cv.resize(source_img, (int(w), int(h)))
        # img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
        img = Image.fromarray(np.uint8(source_img)) # 转换为Image. 
        img = np.array(yolo.detect_image(img))"""

            if res:
                frame_counter += 1
                if frame_counter == int(cap.get(cv.CAP_PROP_FRAME_COUNT)):
                    frame_counter = 0
                    cap.set(cv.CAP_PROP_POS_FRAMES, 0)

                h, w, _ = frame.shape
                detection_img = frame
                if h > 2000 or w > 2000:
                    h = h // 2
                    w = w // 2
                    detection_img = cv.resize(detection_img, (int(w), int(h)))
                detection_img = Image.fromarray(np.uint8(detection_img)) # 转换为Image. 
                detection_img = np.array(yolo.detect_image(detection_img))

                yield cv.imencode('.jpg', np.hstack([frame, detection_img]))[1].tobytes() # to bytes.
            else:
                print("not get cap, please check you camera is opened!")
            
            cv.waitKey(500)





