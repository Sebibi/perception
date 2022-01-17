import numpy as np


def get_vertices(box):
    return np.array([(box[0]+box[3])/2, box[1]])




def get_cone_vertices(boxes):
    return np.apply_along_axis(get_vertices, axis=1, arr=boxes)
