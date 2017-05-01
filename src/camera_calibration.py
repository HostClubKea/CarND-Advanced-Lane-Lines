import numpy as np
import cv2


class CameraCalibration:

    def __init__(self, images, chessboard_size=(9, 6)):
        self.ret = None
        self.mtx = None
        self.dist = None
        self.rvecs = None
        self.tvecs = None
        self.chessboard_size = chessboard_size
        self.found_images = []
        self.unfound_images = []
        self.calibrate(images)

    def calibrate(self, images, draw_corners=True):
        # Prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
        objp = np.zeros((self.chessboard_size[1] * self.chessboard_size[0], 3), np.float32)
        objp[:, :2] = np.mgrid[0:self.chessboard_size[0], 0:self.chessboard_size[1]].T.reshape(-1, 2)

        # Arrays to store object points and image points from all the images.
        objpoints = []
        imgpoints = []

        for img in images:
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

            # Find the chess board corners
            ret, corners = cv2.findChessboardCorners(gray, self.chessboard_size, None)

            # If found, add object points, image points (after refining them)
            if ret:
                objpoints.append(objp)
                imgpoints.append(corners)

                if draw_corners:
                    # Draw the corners
                    self.found_images.append(cv2.drawChessboardCorners(img, self.chessboard_size, corners, ret))
                else:
                    self.found_images.append(img)
            else:
                self.unfound_images.append(img)

        self.ret, self.mtx, self.dist, self.rvecs, self.tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1], None, None)

    def undistort(self, image):
        # Undistort image using computed coefficients
        return cv2.undistort(image, self.mtx, self.dist, None, self.mtx)

