import cv2 as cv
import numpy as np

dist = np.array([-1.50656817e-01, 9.09119885e-02, -3.27882013e-05, -2.35805050e-04, -2.02638144e-02])

# dist = np.array([-1.50023148e-01,  9.02015711e-02, -9.63103978e-05, -2.63814260e-04, -1.98804671e-02])

newmtx = np.array([[1.63746228e+03, 0.00000000e+00, 2.09781369e+03],
          [0.00000000e+00, 1.63423767e+03, 1.08317178e+03],
          [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

# newmtx = np.array([[8.19603210e+02, 0.00000000e+00, 1.04809980e+03],
#                    [0.00000000e+00, 8.18179016e+02, 5.41126793e+02],
#                    [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

mtx = np.array([[1.82045314e+03, 0.00000000e+00, 2.09501251e+03],
       [0.00000000e+00, 1.82052059e+03, 1.08402955e+03],
       [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])


# mtx = np.array([[9.09797410e+02, 0.00000000e+00, 1.04740024e+03],
#                 [0.00000000e+00, 9.09805311e+02, 5.41979594e+02],
#                 [0.00000000e+00, 0.00000000e+00, 1.00000000e+00]])

x, y, w, h = (26, 63, 4050, 2048)
# x, y, w, h = (12, 30, 2026, 1026)

def undist(image):
    dst = cv.undistort(image, mtx, dist, None, newmtx)[x:x + w, y:y + h]
    return dst
