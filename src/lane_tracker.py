import numpy as np
import cv2
from src.lane import Lane

import ctypes

class LaneTracker(object):

    def __init__(self, camera_calibration, perspective_transformer, image_processor,  h=720, w=1280, n_windows=9):

        self.h = h
        self.w = w
        self.n_windows = n_windows
        self.left = None
        self.right = None
        self.camera_calibration = camera_calibration
        self.perspective_transformer = perspective_transformer
        self.image_processor = image_processor
        # Init left and right lanes
        self.left = Lane(self.h, self.w, self.perspective_transformer, self.n_windows,)
        self.right = Lane(self.h, self.w, self.perspective_transformer, self.n_windows)
        self.debug_images=[]

    def process(self, image, force=True, draw_lane=True, draw_debug_info=True):
        # Undistort image
        image = self.camera_calibration.undistort(image)
        # Get edges
        edges = self.image_processor.process(image)
        # Get flatten perspective
        flat_edges = self.perspective_transformer.transform(edges)
        # Get nonzero pixel indices
        nonzero = flat_edges.nonzero()
        # Init lanes
        if len(self.left.windows) == 0 or force:
            # If lanes was'n inited before pass starting evaluated point for the lane
            (x_left, x_right) = self.get_starting_search_points(flat_edges)
            self.left.track(nonzero, x_start=x_left)
            self.right.track(nonzero, x_start=x_right)
        else:
            # Track previously initialized frames
            self.left.track(nonzero)
            self.right.track(nonzero)
            # If lane was'n found in 2 previous frames refresh starting point
            if self.left.not_found > 2 or self.right.not_found > 2:
                (x_left, x_right) = self.get_starting_search_points(flat_edges)
                if self.left.not_found > 2:
                    self.left.track(nonzero, x_start=x_left)

                if self.right.not_found > 2:
                    self.right.track(nonzero, x_start=x_right)

        if draw_debug_info:
            image = self.draw_debug_info(image)

        if draw_lane:
            image = self.draw_lane_overlay(image, unwarp=True)

        return image

    def get_starting_search_points(self, flat_edges):
        # Histogram for bottom part of image
        histogram = np.sum(flat_edges[int(self.h / 2):, :], axis=0)
        # Starting point for left and right lanes
        return (np.argmax(histogram[:self.w // 2]), np.argmax(histogram[self.w // 2:]) + self.w // 2)

    def draw_debug_info(self, image):
        self.debug_images = []
        edges = self.image_processor.process(image, separate_channels=True)
        # Debug image with edges and
        edge_overlay = self.draw_edge_overlay(np.copy(edges))
        self.debug_images.append(edge_overlay)
        edge_overlay = cv2.resize(edge_overlay, (0, 0), fx=0.31, fy=0.31)

        perspective_overlay = self.draw_perspective_overlay(self.perspective_transformer.transform(edges))
        self.debug_images.append(perspective_overlay)
        perspective_overlay = cv2.resize(perspective_overlay, (0, 0), fx=0.31, fy=0.31)

        top_overlay = self.draw_lane_overlay(self.perspective_transformer.transform(image))
        self.debug_images.append(top_overlay)
        top_overlay = cv2.resize(top_overlay, (0, 0), fx=0.31, fy=0.31)


        image[:250, :, :] = image[:250, :, :] * .4
        image[250:450, -450:, :] = image[250:450, -450:, :] * .4
        (h, w, _) = edge_overlay.shape
        image[20:20 + h, 20:20 + w, :] = edge_overlay
        image[20:20 + h, 20 + 20 + w:20 + 20 + w + w, :] = perspective_overlay
        image[20:20 + h, 20 + 20 + 20 + w + w:20 + 20 + 20 + w + w + w, :] = top_overlay

        text_x = 20 + 20 + w + w + 20 + 20
        self.draw_text(image, 'Radius of curvature:  {} m'.format(self.radius_of_curvature()), text_x, h+ 70)
        self.draw_text(image, 'Distance (left):       {:.1f} m'.format(self.left.camera_distance()), text_x, h + 110)
        self.draw_text(image, 'Distance (right):      {:.1f} m'.format(self.right.camera_distance()), text_x, h+ 150)
        self.draw_text(image, 'Not found: right - {} left - {}'.format(self.right.not_found, self.left.not_found), text_x, h + 190)
        return image

    def draw_text(self, frame, text, x, y):
        cv2.putText(frame, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .8, (255, 255, 255), 2)

    def draw_edge_overlay(self, image):
        image = image * 255
        pts = np.array(self.perspective_transformer.source, np.int32)
        cv2.polylines(image, [pts], True, (255, 0, 0), 3)
        return image

    def draw_perspective_overlay(self, binary, lines=True, windows=True):
        if len(binary.shape) == 2:
            image = np.dstack((binary, binary, binary))
        else:
            image = binary
        image = image * 255
        if windows:
            for window in self.left.windows:
                window.draw(image)
            for window in self.right.windows:
                window.draw(image)
        if lines:
            self.left.draw(image)
            self.right.draw(image)

        return image

    def draw_lane_overlay(self, image, unwarp=False):
        # Create an image to draw the lines on
        overlay = np.zeros_like(image).astype(np.uint8)
        points = np.vstack((self.left.get_points(), np.flipud(self.right.get_points())))
        # Draw the lane onto the warped blank image
        cv2.fillPoly(overlay, [points], (0, 255, 0))

        if unwarp:
            # Warp the blank back to original image space using inverse perspective matrix (Minv)
            overlay = self.perspective_transformer.unwarp(overlay)
            # Combine the result with the original image
        return cv2.addWeighted(image, 1, overlay, 0.3, 0)

    def radius_of_curvature(self):
        return int(np.average([self.left.radius_of_curvature(), self.right.radius_of_curvature()]))