import cv2 as cv
import numpy as np
from Api import ApiCamera
import Distortion
from Yolo import Detection, parse_opt
from DepthEstimation import DepthCamera
import KeyPoint
import time
import cv2
from utils.general import check_requirements
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as T
from PIL import Image


class Main:
    api_camera = None
    depth_camera = None
    yolo = None

    def __init__(self):
        camera_position = np.array([0, 0, 0])
        self.api_camera = ApiCamera()
        self.depth_camera = DepthCamera(camera_position)
        self.yolo = Detection()
        image = self.api_camera.get_image()
        opt = parse_opt(image)
        self.yolo.run(**vars(opt))
        boxes, classes = self.yolo.run_inference(**vars(opt))
        # self.api_camera.initstream()

        # cv2.namedWindow('1', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        # cv2.resizeWindow('1', 1200, 600)



    def compute(self, i):
        image_start = time.perf_counter()
        image = self.api_camera.get_image()
        image_end = time.perf_counter()
        # self.api_camera.stream(image)
        # cv2.imshow('1', image)
        # k = cv2.waitKey(1) & 0xFF

        # print(f"Image acquisition time {image_end - image_start}")

        # resize_start = time.perf_counter()
        # img = np.reshape(image, (2176, 4096, -1))
        # small_img = cv.resize(img, dsize=(2409, 1280), interpolation=cv.INTER_CUBIC)
        # resize_end = time.perf_counter()
        # # print(f"Downsample time {resize_end - resize_start}")

        # dist_start = time.perf_counter()
        # undistortedImage = Distortion.undist(image)
        # dist_end = time.perf_counter()
        # print(f"distortion time {dist_end - dist_start}")
        # cv.imwrite("one_image" + str(i) + ".jpg", image)

        yolo_start = time.perf_counter()
        # check_requirements(exclude=('tensorboard', 'thop', 'opencv-python', 'seaborn'))
        opt = parse_opt(image)
        boxes, cone_class = self.yolo.run_inference(**vars(opt))
        yolo_end = time.perf_counter()
        print(f"YOLO Time {yolo_end - yolo_start}")
        # print(cone_class)
        # print(boxes)

        if len(boxes) != 0:
            distance_start = time.perf_counter()
            vertices = KeyPoint.get_cone_vertices(boxes)
            positions = self.depth_camera.compute_distances(vertices)

            # coordinates = [self.depth_camera.get_coordinates(v) for v in vertices]

            # final_cone_class = cone_class[0]  # To remove  dtype=float32 inside the array, [0] on cone class

            distance_end = time.perf_counter()
            print(f"Distance time {distance_end - distance_start}")
            print(f"position : {positions}")
            # print(f"class : {cone_class}")
            # print(f"coordinates : {coordinates}")
            # return positions


def run(iterations):
    program = Main()
    start = time.perf_counter()
    for i in range(iterations):
        program.compute(i)

    end = time.perf_counter()
    print(f"total time for 100 {end - start}")
    print(f"FPS : {nb_photos / (end - start)}")


nb_photos = 100
run(nb_photos)

