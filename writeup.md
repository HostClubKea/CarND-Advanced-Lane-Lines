## Writeup 

#Advanced Lane Finding Project

The goals / steps of this project are the following:

* Compute the camera calibration matrix and distortion coefficients given a set of chessboard images.
* Apply a distortion correction to raw images.
* Use color transforms, gradients, etc., to create a thresholded binary image.
* Apply a perspective transform to rectify binary image ("birds-eye view").
* Detect lane pixels and fit to find the lane boundary.
* Determine the curvature of the lane and vehicle position with respect to center.
* Warp the detected lane boundaries back onto the original image.
* Output visual display of the lane boundaries and numerical estimation of lane curvature and vehicle position.

[//]: # (Image References)

[image1]: ./examples/calibration/calibration_13.png "Calibration"
[image2]: ./examples/calibration/calibration_undistorted_13.png "Undistorted"

[image3]: ./test_images/straight_lines1.jpg "Undistorted"
[image4]: ./examples/undistort/straight_lines1.jpg.png "Undistorted"

[image5]: ./test_images/test3.jpg "Road Transformed"
[image6]: ./examples/edges/separate_test3.jpg.png "Binary Example"
[image7]: ./examples/perspective/straight_lines1.jpg.png "Warp Example"
[image8]: ./examples/perspective/transformed_straight_lines1.jpg.png "Fit Visual"
[image9]: ./examples/output/test3.jpg_warp_window.png "Output"
[image10]: ./examples/perspective/transformed_test3.jpg.png "Output"
[image11]: ./output_images/test3.jpg.png "Output"

[video10]: ./project_video.mp4 "Video"

### Camera Calibration


The code for this step is contained in the `CameraCalibration` class located in "./src/camera_calibration.py". Constractor of this class accepts callibration images and chessboard shape and calculate distortion coefficients out of them. 
I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image. Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

Then the output `objpoints` and `imgpoints` are used to compute the camera calibration and distortion coefficients. 

**Original calibration image**

![alt text][image1]

**Undistorted calibration image**

![alt text][image2]

### Pipeline (single images)

#### 1. Distortion correction.

As camera calibration object was already created we can use stored distortion coefficients and undistort road image. 

**Original road image**

![alt text][image3]

**Undistorted road image**

![alt text][image4]

#### 2. Edge detection.

I used a combination of color and gradient thresholds to generate a binary image. Helper function for color and gradient threshhold could be found in 
`src/gradients.py`

Color and gradient detection is dane on V channel of HSV color scheme. It seems to work well on wide range of light image conditions

### Gradient absolute value
For absolute gradient value we simply apply a threshold to `cv2.Sobel()` output for each axis.

```python
gradient_x = abs_sobel_threshold(v_channel, orient='x', sobel_kernel=3, thresh=(20, 100))
gradient_y = abs_sobel_threshold(v_channel, orient='y', sobel_kernel=3, thresh=(20, 100))
```

### Gradient magnitude
Include pixels within a threshold of the gradient magnitude.

```python
magnitude = mag_threshold(v_channel, sobel_kernel=3, thresh=(20, 100))
```

### Gradient direction
Include pixels that happen to be withing a threshold of the gradient direction.

```python
direction = dir_threshold(v_channel, sobel_kernel=3, thresh=(0.7, 1.3))
```

### Combining gradients
Include pixels that happen to be withing a threshold of the gradient direction.

```python
gradient_mask = np.zeros_like(v_channel)
gradient_mask[((gradient_x == 1) & (gradient_y == 1)) | ((magnitude == 1) & (direction == 1))] = 1
```


### Color
Also threshold is applied on absolute value of V chanel.

```python
color_mask = color_threshold(v_channel, thresh=(220, 255))
```

![alt text][image6]

Some source showed good results by separate tracking white and yellow lanes which might be direction to improve lane detection results

#### 3. Perspective transformation

The code for my perspective transform could be found in the `PerspectiveTransformer` class located in "./src/perspective_transformer.py". It accepts array of source and destination points and then use them in `transform` and `unwarp` functions. 

Initializing of source and destionation points could be found in `P4.py` (P4.py):

```
source = np.float32([[w // 2 - 67, h * .625], [0, h], [w, h], [w // 2 + 67, h * .625]])
```
```
destination = np.float32([[100, 0], [100, h], [w - 100, h], [w - 100, 0]])
```

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

![alt text][image7]

![alt text][image8]

#### 4. Detecting Lines
Detecting lines code splited on 3 classes `LaneTracker`, `Lane` and `Window` located in "./src/lane_tracker.py", "./src/lane.py", "./src/window.py" 

First we need to find starting points to the lines. It could be done based on histogram of edge points for the bottom half of flattened edge image.
This starting points passed to `Lane` object which would start tracking road lane based on sliding window algorithm - we scan the frame with  windows, collecting non-zero pixels within window bounds. Once top is reached, we fit a second order polynomial into collected points. Then this polynomial coefficients would represent a single lane boundary.

Lane is smoothed using previous lanes coefficients to avoid jumps and wobbling. 

#####Road image after perspective transformation
![alt text][image10]

#####Windows and lanes on edge image:

![alt text][image9]

#### 5. Calculating the radius of curvature of the lane and the position of the vehicle with respect to center.

To calculate lane curvature we first need to find relation between picture and real world dimensions. Knowing US lane standarts - line height 3 meters and distance betwenn 2 lines 3.7 meters we get next ratios:

Meters per pixel in x dimension
```
xm_per_pix = 3.7 / 750
```

Meters per pixel in y dimension
```
ym_per_pix = 3. / 80
```

Then using formulas for radius curvature from [there](http://www.intmath.com/applications-differentiation/8-radius-curvature.php) and founded polyline coefficients for te lanes I calculate road curvature

Code could be found in "./src/lane.py" in `radius_of_curvature` function


Vehicle position also could be approximate within the lane. It's a distance between middle of the frame and position of the lane. 

Code could be found in "./src/lane.py" in `camera_distance` function

#### 6. Output example.

Pipeline for tracking lanes implemented in `src/lane_tracker.py` in the function `process()` . Here is an example of my result on a test image:

![alt text][image11]

---

### Pipeline (video)

Resulting video could be found in `output_video` folder 

---

### Discussion

#### 1. Problem and fails

Running project on advance challenge video shows many weekness and problems of current pipeline:

1) Uphills and downhills change perspective 

2) Perspective doesn't work well with fast turns

3) Fixed threshholds doesn't work well with changing light conditions

4) Bright sun spots and dark shadows make finding lane 

Trying to work on harder challenge video make me think that probably pipeline should be improved with some probabalistic model and dynamic thresholds. We can find lane in good light condition parts and then use this information to change threshholds some way to maximize points in lane area and minimize on the whole image 
