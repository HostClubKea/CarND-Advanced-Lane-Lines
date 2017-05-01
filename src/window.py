import numpy as np
import cv2


class Window(object):

    def __init__(self, y1, y2, x, m=100, tolerance=20):

        self.x = x
        self.mean_x = x
        self.y1 = y1
        self.y2 = y2
        self.m = m
        self.tolerance = tolerance

    def pixels(self, nonzero, x=None):
        if x is not None:
            self.x = x

        indices = (
            (nonzero[0] >= self.y1) & (nonzero[0] < self.y2) &
            (nonzero[1] >= self.x - self.m) & (nonzero[1] < self.x + self.m)
        ).nonzero()[0]

        if len(indices) > self.tolerance:
            self.mean_x = np.int(np.mean(nonzero[1][indices]))
        else:
            self.mean_x = self.x

        return indices

    def draw(self, image):
        cv2.rectangle(image, (self.x - self.m, self.y1), (self.x + self.m, self.y2), (255, 255, 0), 2)