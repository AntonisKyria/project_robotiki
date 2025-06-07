import numpy as np
import cv2

def normalize(v):
    norm = np.linalg.norm(v)
    return v / norm if norm != 0 else v

def draw_axis(image, origin, x_axis, y_axis, z_axis, scale=100):
    origin = tuple(np.int32(origin))
    x_end = tuple(np.int32(origin + scale * x_axis))
    y_end = tuple(np.int32(origin + scale * y_axis))
    z_end = tuple(np.int32(origin + scale * z_axis))
    cv2.line(image, origin, x_end, (0, 0, 255), 3)  # X - RED
    cv2.line(image, origin, y_end, (0, 255, 0), 3)  # Y - GREEN
    cv2.line(image, origin, z_end, (255, 0, 0), 3)  # Z - BLUE
