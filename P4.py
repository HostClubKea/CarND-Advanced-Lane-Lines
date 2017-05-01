import os

import matplotlib.image as mpimg
import numpy as np

def get_images(dir):
    files = os.listdir(dir)
    images = []
    names = []
    for file in files:
        image = mpimg.imread(os.path.join(dir, file))
        images.append(image)
        names.append(file)
    return images, names

from src.camera_calibration import CameraCalibration

calibration_images, _ = get_images('camera_cal')
camera_calibration = CameraCalibration(calibration_images)

# Save undistorted calibration images
if False:
    i = 0
    for image in calibration_images:
        mpimg.imsave("examples/calibration/calibration_{}.png".format(i), image)
        image = camera_calibration.undistort(image)
        mpimg.imsave("examples/calibration/calibration_undistorted_{}.png".format(i), image)
        i+=1

test_images, test_names = get_images('test_images')
if False:
    for i in range(len(test_images)):
        test_images[i] = camera_calibration.undistort(test_images[i])
        mpimg.imsave("examples/undistort/{}.png".format(test_names[i]),  test_images[i])


from src.perspective_transformer import PerspectiveTransformer
import cv2

w = 1280
h = 720
source = np.float32([[w // 2 - 67, h * .625], [0, h], [w, h], [w // 2 + 67, h * .625]])
destination = np.float32([[100, 0], [100, h], [w - 100, h], [w - 100, 0]])
# Meters per pixel in y dimension
ym_per_pix = 3. / 80
# Meters per pixel in x dimension
xm_per_pix = 3.7 / 750

perspective_transformer = PerspectiveTransformer(source, destination, xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix)
if False:
    for i in range(len(test_images)):
        image = np.copy(test_images[i])
        pts = np.array(perspective_transformer.source, np.int32)
        cv2.polylines(image, [pts], True, (255, 0, 0), 3)

        mpimg.imsave("examples/perspective/{}.png".format(test_names[i]), image)

        image = perspective_transformer.transform(test_images[i])
        pts = np.array(perspective_transformer.destination, np.int32)
        cv2.polylines(image, [pts], True, (255, 0, 0), 3)

        mpimg.imsave("examples/perspective/transformed_{}.png".format(test_names[i]), image)

from src.gradient import *
from skimage import exposure
import ctypes

import matplotlib.pyplot as plt

def draw_channel(image, w_image, channel_image, channels = ['R', 'G', 'B'], color_thresh=[(170,255), (170, 255), (170, 255)], name = "examples/thresholds/rgb.png"):

    f, ((ax1, ax2, ax3), (ax4, ax5, ax6), (ax7, ax8, ax9), (ax10, ax11, ax12)) = plt.subplots(4, 3, sharey='col', sharex='row', figsize=(10, 8))
    f.tight_layout()

    ax1.set_title('Original Image', fontsize=16)
    ax1.imshow(image)

    ax2.set_title('Warped Image', fontsize=16)
    ax2.imshow(w_image)

    c1 = channel_image[:, :, 0]
    # c1 = c1/np.amax(c1)*255
    ax3.set_title('{} , min={}, max={}'.format(channels[0], np.amin(c1), np.amax(c1)), fontsize=16)
    ax3.imshow(c1, cmap='gray')

    c2 = channel_image[:, :, 1]
    # c2 = c2/np.amax(c2)*255
    ax4.set_title('{} , min={}, max={}'.format(channels[1], np.amin(c1), np.amax(c1)), fontsize=16)
    ax4.imshow(c2, cmap='gray')

    c3 = channel_image[:, :, 2]
    # c3 = c3/np.amax(c3)*255
    ax5.set_title('{} , min={}, max={}'.format(channels[2], np.amin(c1), np.amax(c1)), fontsize=16)
    ax5.imshow(c3, cmap='gray')


    t1 = color_threshold(c1, thresh=color_thresh[0])
    ax6.set_title('{} , min={}, max={}'.format(channels[0], color_thresh[0][0], color_thresh[0][1]), fontsize=16)
    ax6.imshow(t1, cmap='gray')

    t2 = color_threshold(c2, thresh=color_thresh[1])
    ax7.set_title('{} , min={}, max={}'.format(channels[1], color_thresh[1][0], color_thresh[1][1]), fontsize=16)
    ax7.imshow(t2, cmap='gray')

    t3 = color_threshold(c3, thresh=color_thresh[2])
    ax8.set_title('{} , min={}, max={}'.format(channels[2], color_thresh[2][0], color_thresh[2][1]), fontsize=16)
    ax8.imshow(t3, cmap='gray')

    # ch1 = cv2.equalizeHist(c1)
    # # c1 = c1/np.amax(c1)*255
    # ax9.set_title('{} , min={}, max={}'.format(channels[0], np.amin(ch1), np.amax(ch1)), fontsize=16)
    # ax9.imshow(ch1, cmap='gray')
    #
    # ch2 = cv2.equalizeHist(c2)
    # # c2 = c2/np.amax(c2)*255
    # ax10.set_title('{} , min={}, max={}'.format(channels[1], np.amin(ch2), np.amax(ch2)), fontsize=16)
    # ax10.imshow(ch2, cmap='gray')
    #
    # ch3 = cv2.equalizeHist(c3)
    # # c3 = c3/np.amax(c3)*255
    # ax11.set_title('{} , min={}, max={}'.format(channels[2], np.amin(ch3), np.amax(ch3)), fontsize=16)
    # ax11.imshow(ch3, cmap='gray')


    plt.savefig(name)

#Display different threshholds
if False:
    for i in range(len(test_images)):
        image = np.copy(test_images[i])

        image = camera_calibration.undistort(image)
        # image = (image / 255.).astype(np.float32)
        # image = exposure.equalize_adapthist(image)
        # image = (image * 255).astype(ctypes.c_ubyte)
        image = cv2.GaussianBlur(image, (5, 5), 0)

        w_image = perspective_transformer.transform(image)

        hsv = cv2.cvtColor(w_image, cv2.COLOR_RGB2HSV)
        lab = cv2.cvtColor(w_image, cv2.COLOR_RGB2LAB)
        luv = cv2.cvtColor(w_image, cv2.COLOR_RGB2LUV)
        hls = cv2.cvtColor(w_image, cv2.COLOR_RGB2HLS)
        yCrCb = cv2.cvtColor(w_image, cv2.COLOR_RGB2YCrCb)

        draw_channel(image, w_image, w_image, name="examples/thresholds/{}_rgb.png".format(test_names[i]))
        draw_channel(image, w_image, yCrCb, channels=['Y', 'Cr', 'Cb'], name="examples/thresholds/{}_yCrCb.png".format(test_names[i]))
        draw_channel(image, w_image, hls, channels=['H', 'L', 'S'], name="examples/thresholds/{}_hls.png".format(test_names[i]))
        draw_channel(image, w_image, luv, channels=['L', 'U', 'V'], name="examples/thresholds/{}_luv.png".format(test_names[i]))
        draw_channel(image, w_image, lab, channels=['L', 'A', 'B'], name="examples/thresholds/{}_lab.png".format(test_names[i]))
        draw_channel(image, w_image, hsv, channels=['H', 'S', 'V'], name="examples/thresholds/{}_hsv.png".format(test_names[i]), color_thresh=[(170,255), (170, 255), (220, 255)])

    pass


image_processor = ImageProcessor()

if False:
    for i in range(len(test_images)):
        image = np.copy(test_images[i])

        image_separate = image_processor.process(image, separate_channels=True)
        image_binary = image_processor.process(image, separate_channels=False)

        mpimg.imsave("examples/edges/separate_{}.png".format(test_names[i]), image_separate)
        mpimg.imsave("examples/edges/{}.png".format(test_names[i]), image_binary)

from src.lane_tracker import  LaneTracker

if False:
    for i in range(len(test_images)):
        image = np.copy(test_images[i])

        lane_tracker = LaneTracker(camera_calibration, perspective_transformer, image_processor, n_windows=18)
        image = lane_tracker.process(image, draw_lane=True, draw_debug_info=True)
        mpimg.imsave("output_images/{}.png".format(test_names[i]), image)

        lane_tracker.debug_images[0] = cv2.cvtColor(lane_tracker.debug_images[0].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite("examples/output/{}_binary.png".format(test_names[i]), lane_tracker.debug_images[0])
        lane_tracker.debug_images[1] = cv2.cvtColor(lane_tracker.debug_images[1].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite("examples/output/{}_warp_window.png".format(test_names[i]), lane_tracker.debug_images[1])
        lane_tracker.debug_images[2] = cv2.cvtColor(lane_tracker.debug_images[2].astype(np.uint8), cv2.COLOR_RGB2BGR)
        cv2.imwrite("examples/output/{}_warp_lane.png".format(test_names[i]), lane_tracker.debug_images[2])


from moviepy.editor import VideoFileClip

if False:
    video_output_name = 'project_video_annotated.mp4'
    video = VideoFileClip("project_video.mp4")
    tracker = LaneTracker(camera_calibration, perspective_transformer, image_processor, n_windows=18)
    video_output = video.fl_image(tracker.process)
    video_output.write_videofile(video_output_name, audio=False)



source = np.float32([[40, 720], [490, 482], [810, 482], [1250, 720]])
destination = np.float32([[40, 720], [0, 0], [1280, 1], [1250, 720]])
perspective_transformer = PerspectiveTransformer(source, destination, xm_per_pix=xm_per_pix, ym_per_pix=ym_per_pix)

if True:
    video_output_name = 'challenge_video_annotated.mp4'
    video = VideoFileClip("challenge_video.mp4")
    tracker = LaneTracker(camera_calibration, perspective_transformer, image_processor, n_windows=18)
    video_output = video.fl_image(tracker.process)
    video_output.write_videofile(video_output_name, audio=False)