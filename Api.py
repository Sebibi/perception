from __future__ import print_function
import os
import platform
import sys
from mvIMPACT import acquire
from mvIMPACT.Common import exampleHelper


import numpy as np
import ctypes
import cv2


class ApiCamera:
    factor = 4
    h = int(2176 / factor)
    w = int(4096 / factor)
    device_manager = None
    device = None
    fi = None
    statistics = None
    previous_request = None
    acquisition_ctrl = None

    def __init__(self):
        self.device_manager = acquire.DeviceManager()
        self.device = self.device_manager.getDevice(0)
        self.device.open()
        self.fi = acquire.FunctionInterface(self.device)
        self.fi.loadSetting("camera_settings/high_speed", acquire.sfFile)
        self.formatControl = acquire.ImageFormatControl(self.device)
        self.formatControl.decimationHorizontalMode.write(1)
        self.formatControl.decimationVerticalMode.write(1)
        self.formatControl.decimationVertical.write(self.factor)
        self.formatControl.decimationHorizontal.write(self.factor)

        self.acquisition_ctrl = acquire.AcquisitionControl(self.device)
        self.acquisition_ctrl.acquisitionFrameRateEnable.write(True)
        self.acquisition_ctrl.acquisitionFrameRate.write(10)        

        

        self.statistics = acquire.Statistics(self.device)

        buffer_size = 0
        while self.fi.imageRequestSingle() == acquire.DMR_NO_ERROR:
            buffer_size += 1
            print("Buffer queued")
        self.previous_request = None

        exampleHelper.manuallyStartAcquisitionIfNeeded(self.device, self.fi)

    def acquire_request(self):
        self.fi.imageRequestSingle()
        requestNr = self.fi.imageRequestWaitFor(10000)
        if self.fi.isRequestNrValid(requestNr):
            pRequest = self.fi.getRequest(requestNr)
            if pRequest.isOK:
                return pRequest
        else:
            print("imageRequestWaitFor failed (" + str(
                requestNr) + ", " + acquire.ImpactAcquireException.getErrorCodeAsString(requestNr) + ")")
        return None

    def to_text(self, pRequest):
        cbuf = (ctypes.c_char * pRequest.imageSize.read()).from_address(int(pRequest.imageData.read()))
       # arr = np.frombuffer(cbuf, dtype=np.uint8).reshape(1088, 2048, 4)
        arr = np.frombuffer(cbuf, dtype=np.uint8).reshape(self.h, self.w, 4)
        return arr[:, :, :-1]

    def unlock_buffer(self, pRequest):
        if self.previous_request is not None:
            self.previous_request.unlock()
        self.previous_request = pRequest
        # self.fi.imageRequestReset()
        # self.fi.imageRequestSingle()

    def print_statistic(self, i):
        if i % 100 == 0:
            print("Info from " + self.device.serial.read() +
                  ": " + self.statistics.framesPerSecond.name() + ": " + self.statistics.framesPerSecond.readS() +
                  ", " + self.statistics.errorCount.name() + ": " + self.statistics.errorCount.readS() +
                  ", " + self.statistics.captureTime_s.name() + ": " + self.statistics.captureTime_s.readS())

    def continuous_display(self, frames):
        display_window = acquire.ImageDisplayWindow("A window created from Python")
        display = display_window.GetImageDisplay()
        for i in range(frames):
            pRequest = self.acquire_request()
            display.SetImage(pRequest)
            display.Update()
            self.print_statistic(i)
            self.unlock_buffer(pRequest)
        return

    def get_image(self):
        request = self.acquire_request()
        print(request.infoTimeStamp_us.read() / 1000000.0)
        image = self.to_text(request)
        self.unlock_buffer(request)
        self.print_statistic(0)
        return image


    def stream(self, image):
        cv2.imshow('1', image)

    
    def initstream(self):
        cv2.namedWindow('1', cv2.WINDOW_NORMAL | cv2.WINDOW_GUI_NORMAL)
        cv2.resizeWindow('1', 1200, 600)









