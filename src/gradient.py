import numpy as np
import cv2

def abs_sobel_threshold(image, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    if orient == 'x':
        sobel = cv2.Sobel(image, cv2.CV_64F, 1, 0)
    else:
        sobel = cv2.Sobel(image, cv2.CV_64F, 0, 1)
    abs_sobel = np.absolute(sobel)

    scaled_sobel = np.uint8(255 * abs_sobel / np.max(abs_sobel))

    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    return grad_binary

def mag_threshold(image, sobel_kernel=3, thresh=(0, 255)):
    # Calculate gradient magnitude
    #Take the gradient in x and y separately
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Calculate the magnitude
    gradmag = np.sqrt(sobelx ** 2 + sobely ** 2)
    # Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmag) / 255
    gradmag = (gradmag / scale_factor).astype(np.uint8)
    # Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(gradmag)
    mag_binary[(gradmag >= thresh[0]) & (gradmag <= thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Take the gradient in x and y separately
    sobelx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=sobel_kernel)
    sobely = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=sobel_kernel)
    # Take the absolute value of the x and y gradients
    abs_sobelx = np.absolute(sobelx)
    abs_sobely = np.absolute(sobely)
    # Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    absgraddir = np.arctan2(abs_sobely, abs_sobelx)
    # Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(absgraddir)
    dir_binary[(absgraddir >= thresh[0]) & (absgraddir <= thresh[1])] = 1
    return dir_binary


def color_threshold(image, thresh=(0, 255)):
    mask = np.zeros_like(image)
    mask[(image >= thresh[0]) & (image <= thresh[1])] = 1
    return mask

from skimage import exposure
import ctypes

class ImageProcessor():
    def __init__(self):
        self.channel_order = 'RGB'

    def process(self, image, separate_channels = False):
        frame = cv2.GaussianBlur(image, (5, 5), 0)
        frame = (image / 255.).astype(np.float32)
        frame = exposure.equalize_adapthist(frame)
        frame = (frame * 255).astype(ctypes.c_ubyte)
        frame = cv2.GaussianBlur(frame, (5, 5), 0)

        hsv = cv2.cvtColor(np.copy(frame), cv2.COLOR_RGB2HSV).astype(np.float)
        v_channel = hsv[:, :, 2]

        # Get a combination of all gradient thresholding masks
        gradient_x = abs_sobel_threshold(v_channel, orient='x', sobel_kernel=3, thresh=(20, 100))
        gradient_y = abs_sobel_threshold(v_channel, orient='y', sobel_kernel=3, thresh=(20, 100))
        magnitude = mag_threshold(v_channel, sobel_kernel=3, thresh=(20, 100))
        direction = dir_threshold(v_channel, sobel_kernel=3, thresh=(0.7, 1.3))
        gradient_mask = np.zeros_like(v_channel)
        gradient_mask[((gradient_x == 1) & (gradient_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
        # Get a color thresholding mask


        color_mask = color_threshold(v_channel, thresh=(220, 255))
        if separate_channels:
            return np.dstack((np.zeros_like(v_channel), gradient_mask, color_mask))
        else:
            mask = np.zeros_like(gradient_mask)
            mask[(gradient_mask == 1) | (color_mask == 1)] = 1
            return mask






# Define a function that thresholds the S-channel of HLS
def hls_select(img, channel_order="BGR", thresh=(90, 255)):
    if channel_order == "RGB":
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    else:
        hls = cv2.cvtColor(img, cv2.COLOR_BGR2HLS)
    s_channel = hls[:,:,2]
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output


def hsv_select(img, channel_order="BGR", thresh_H=(0, 255), thresh_S=(0, 255), thresh_V=(0, 255)):
    if channel_order == "RGB":
        hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    else:
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    H_channel = hsv[:, :, 0]
    S_channel = hsv[:, :, 1]
    V_channel = hsv[:, :, 2]

    condition_H = (H_channel >= thresh_H[0]) & (H_channel <= thresh_H[1])
    condition_S = (S_channel >= thresh_S[0]) & (S_channel <= thresh_S[1])
    while True:
        binary_output = np.zeros_like(H_channel)
        condition_V = (V_channel >= thresh_V[0]) & (V_channel <= thresh_V[1])
        binary_output[condition_H & condition_S & condition_V] = 1
        break
        # print(thresh_V)
        # len_ = len(binary_output.nonzero()[0])
        # print(len_)
        # if len_ > 10000 and thresh_V[0] != thresh_V[1]:
        #
        #     thresh_V1 = (thresh_V[0], thresh_V[1] - 5)
        #
        #     binary_output = np.zeros_like(H_channel)
        #     condition_V = (V_channel >= thresh_V1[0]) & (V_channel <= thresh_V1[1])
        #     binary_output[condition_H & condition_S & condition_V] = 1
        #     len1 = len(binary_output.nonzero()[0])
        #
        #     thresh_V2 = (thresh_V[0]+5, thresh_V[1])
        #
        #     binary_output = np.zeros_like(H_channel)
        #     condition_V = (V_channel >= thresh_V2[0]) & (V_channel <= thresh_V2[1])
        #     binary_output[condition_H & condition_S & condition_V] = 1
        #     len2 = len(binary_output.nonzero()[0])
        #
        #     if len_ - len2 > len_ - len1:
        #         thresh_V = thresh_V2
        #     else:
        #         thresh_V = thresh_V1
        # else:
        #     break

    return binary_output