import numpy as np
import cv2
from collections import deque
from src.window import Window

class Lane(object):

    def __init__(self, h, w, perspective_transformer,n_windows = 9):
        # Image height in pixels
        self.h = h
        # Image width in pixel
        self.w = w
        # List of recent polynomial coefficients
        self.coefficients = deque(maxlen=5)
        # List of recent polynomial coefficients
        self.last_coefficients = None
        # Amount of not found lanes in recent frames
        self.not_found = 0
        # Windows amount
        self.n_windows = n_windows
        # Window height
        self.window_height = int(self.h / self.n_windows)
        # List of searching windows
        self.windows = []
        self.perspective_transformer = perspective_transformer

    def track(self, nonzero, x_start = None):
        if x_start is not None:
            indices = self.init_windows(nonzero, x_start)
        else:
            indices = self.scan_windows(nonzero)

        self.process_points(nonzero[1][indices], nonzero[0][indices])

    def scan_windows(self, nonzero):
        indices = np.empty([0], dtype=np.int)
        window_x = None
        for window in self.windows:
            indices = np.append(indices, window.pixels_in(nonzero, window_x), axis=0)
            window_x = window.mean_x
        return indices

    def init_windows(self, nonzero, x_start):
        indices = np.empty([0], dtype=np.int)
        self.windows = []
        for i in range(self.n_windows):
            window = Window(
                y1=self.h - (i + 1) * self.window_height,
                y2=self.h - i * self.window_height,
                x=self.windows[-1].mean_x if len(self.windows) > 0 else x_start
            )
            indices = np.append(indices, window.pixels(nonzero), axis=0)
            self.windows.append(window)
        return indices

    def is_good_lane(self, x, y):
        enough_points = len(y) > 0 and np.max(y) - np.min(y) > self.h * .625
        return enough_points

    def process_points(self, x, y):
        if self.is_good_lane(x, y) or len(self.coefficients) == 0:
            self.fit(x, y)
            self.not_found = 0
        else:
            self.not_found += 1

    def get_points(self):
        y = np.linspace(0, self.h - 1, self.h)
        current_fit = self.averaged_fit()
        return np.stack((
            current_fit[0] * y ** 2 + current_fit[1] * y + current_fit[2],
            y
        )).astype(np.int).T

    def averaged_fit(self):
        return np.array(self.coefficients).mean(axis=0)

    def fit(self, x, y):
        if len(x) != 0:
            self.last_coefficients = np.polyfit(y, x, 2)
            self.coefficients.append(self.last_coefficients)
        else:
            self.last_coefficients = None

    def radius_of_curvature(self):
        points = self.get_points()
        y = points[:, 1]
        x = points[:, 0]
        fit_cr = np.polyfit(y * self.perspective_transformer.ym_per_pix, x * self.perspective_transformer.xm_per_pix, 2)
        # Estimate radius of curvature in meters.
        return int(((1 + (2 * fit_cr[0] * 720 * self.perspective_transformer.ym_per_pix + fit_cr[1]) ** 2) ** 1.5) / np.absolute(2 * fit_cr[0]))

    def camera_distance(self):
        # Estimated distance to camera in meters.
        points = self.get_points()
        x = points[np.max(points[:, 1])][0]
        return np.absolute((self.w // 2 - x) * self.perspective_transformer.xm_per_pix)



    def draw(self, image):
        cv2.polylines(image, [self.get_points()], False, (255, 0, 0), 2)