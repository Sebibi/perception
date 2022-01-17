import camera
import numpy as np


class DepthCamera:
    camera = None
    translation = None

    def __init__(self, camera_position):
        self.camera = camera.Camera()
        self.camera.load("camera_parameters/camera_calib")
        self.camera.set_t(camera_position[np.newaxis].T)
        self.translation = camera_position

    def get_coordinates(self, pixel):
        height = 30.6
        z_proj = 200.  # arbitrary
        pixel_coord = np.array([pixel[0], pixel[1], 1.]).reshape(3, 1)
        # get projection
        cone_world = self.camera.image_to_world(pixel_coord, z_proj).reshape(-1)

        x_cam = self.translation[0]
        y_cam = self.translation[1]
        z_cam = self.translation[2]

        # compute cone position (x and depth)
        x_proj = cone_world[0]
        y_proj = cone_world[1]

        alpha = (y_cam - height) / (y_cam + y_proj)

        x_pred = -x_cam + (x_proj + x_cam) * alpha
        z_pred = -z_cam + (z_proj + z_cam) * alpha

        cone_position = np.array([x_pred, height, z_pred])
        return cone_position

    def compute_distance(self, pixel):
        height = 30.6
        z_proj = 200.  # arbitrary
        pixel_coord = np.array([pixel[0], pixel[1], 1.]).reshape(3, 1)
        # get projection
        cone_world = self.camera.image_to_world(pixel_coord, z_proj).reshape(-1)

        x_cam = self.translation[0]
        y_cam = self.translation[1]
        z_cam = self.translation[2]

        # compute cone position (x and depth)
        x_proj = cone_world[0]
        y_proj = cone_world[1]

        alpha = (y_cam - height) / (y_cam + y_proj)

        x_pred = -x_cam + (x_proj + x_cam) * alpha
        z_pred = -z_cam + (z_proj + z_cam) * alpha

        cone_position = np.array([x_pred,height, z_pred])
        dist = np.linalg.norm(cone_position)
        return dist

    def compute_distances(self, pixels):
        return np.apply_along_axis(self.compute_distance, axis=1, arr=pixels)


