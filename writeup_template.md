## Writeup Template
### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Advanced Lane Finding Project**

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

[image0]: ./camera_cal/calibration1.jpg "Distorted"
[image1]: ./output_images/undistorted_calibration1.jpg "Undistorted"
[image2]: ./test_images/test1.jpg "Raw Road Image"
[image3]: ./output_images/undistorted_test1.jpg "Undistorted Road Image"
[image4]: ./output_images/thresholded_test1.png "Thresholded Binary Image"
[image5]: ./test_images/straight_lines1.jpg "Unwarped Straight Line"
[image6]: ./output_images/warped_straight_lines1.jpg "Warped Straight Line"
[image7]: ./test_images/test2.jpg "Unfit Visual"
[image8]: ./output_images/poly_test2.jpg "Fit Visual"
[image9]: ./test_images/test3.jpg "Output"
[image10]: ./output_images/aug_test3.jpg "Output"
[video1]: ./project_video.mp4 "Video"

## [Rubric](https://review.udacity.com/#!/rubrics/571/view) Points
### Here I will consider the rubric points individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Advanced-Lane-Lines/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!
### Camera Calibration

#### 1. Briefly state how you computed the camera matrix and distortion coefficients. Provide an example of a distortion corrected calibration image.

The code for this step is contained in lines 7 through 46 of the file called `calibrate_undistort.py`).  

I start by preparing "object points", which will be the (x, y, z) coordinates of the chessboard corners in the world. Here I am assuming the chessboard is fixed on the (x, y) plane at z=0, such that the object points are the same for each calibration image.  Thus, `objp` is just a replicated array of coordinates, and `objpoints` will be appended with a copy of it every time I successfully detect all chessboard corners in a test image.  `imgpoints` will be appended with the (x, y) pixel position of each of the corners in the image plane with each successful chessboard detection.  

I then used the output `objpoints` and `imgpoints` to compute the camera calibration and distortion coefficients using the `cv2.calibrateCamera()` function.  I applied this distortion correction to the test image using the `cv2.undistort()` function and obtained this result: 

Distorted                  |  Undistorted
:-------------------------:|:-------------------------:
![alt_text][image0]        |  ![alt_text][image1]

### Pipeline (single images)

#### 1. Provide an example of a distortion-corrected image.
To demonstrate this step, I will describe how I apply the distortion correction to one of the test images like this one:

Distorted                  |  Undistorted
:-------------------------:|:-------------------------:
![alt_text][image2]        |  ![alt_text][image3]

#### 2. Describe how (and identify where in your code) you used color transforms, gradients or other methods to create a thresholded binary image.  Provide an example of a binary image result.
I used a combination of color and gradient thresholds to generate a binary image (thresholding steps at lines 36 through 110 and lines 129 through 156 in `find_lane.py`).  Four different kinds of gradient related thresholds are applied onto the grayscaled image, and then the S channel of HLS color space of the image is extracted out. The thresholded grayscaled image and the S channel are then fused to obtain the final result. The tuning of various threshold parameters is the key to the
success of identifying the lanes with less noise. Here's an example of my output for this step.  

Unthresholded              | Thresholded 
:-------------------------:|:-------------------------:
![alt_text][image2]        |  ![alt_text][image4]


#### 3. Describe how (and identify where in your code) you performed a perspective transform and provide an example of a transformed image.

The code for my perspective transform includes a function called `perspective_transform()`, which appears in lines 186 through 221 in the file `find_lane.py`.  The `perspective_transform()` function takes as inputs an image (`img`), as well as source (`src`), (`mtx`) and (`dist`). The destination (`dst`) points are hard coded in the body of the `perspective_transform()` function.  The source and destination points in the following manner:

```
src = [[570,460],[720,460],[1170,720],[160,720]]
offset = 100
dst = np.float32([[offset, offset1],
                  [img_size[0]-offset, offset1],
                  [img_size[0]-offset, img_size[1]],
                  [offset, img_size[1]]])

```
This resulted in the following source and destination points:

| Source        | Destination   | 
|:-------------:|:-------------:| 
| 570, 460      | 100, 0        | 
| 720, 460      | 1180,0        |
| 1170, 720     | 1180, 720     |
| 160, 720      | 100, 720      |

I verified that my perspective transform was working as expected by drawing the `src` and `dst` points onto a test image and its warped counterpart to verify that the lines appear parallel in the warped image.

Unwarped                   | Warped 
:-------------------------:|:-------------------------:
![alt_text][image5]        |  ![alt_text][image6]

#### 4. Describe how (and identify where in your code) you identified lane-line pixels and fit their positions with a polynomial?

After obtained the grayscaled images of the lanes, a histogram of the image is computed describing the distribution of the white pixels along the x-axis. This was performed not for the entire image, but for each sliding window along the y axis. The peak positions can be obtained from the series of histograms and the adjacent white pixels were picked out to generate the final "pixelized" lanes. A 2nd order polynomial fitting was exerted on the lanes to get the curve representation of the left
and right lane. The fitted lane lines with a 2nd order polynomial kinda like this:

Raw                        | Fitted 
:-------------------------:|:-------------------------:
![alt_text][image7]        |  ![alt_text][image8]

#### 5. Describe how (and identify where in your code) you calculated the radius of curvature of the lane and the position of the vehicle with respect to center.

I did this in lines 282 through 284 (shift with respect to center) and lines 177 through 188 (curvature) in my code in `find_lane_pipeline.py`. Basically, after obtaining the two polynomial fitted lines for the left and right lanes in the warped space, the lines are scaled to their dimensions in the physical world and the radii are calculated for each of the line using the curvature formula. Then the final radius is computed as the average of the two radii. 

#### 6. Provide an example image of your result plotted back down onto the road such that the lane area is identified clearly.

I implemented this step in lines 219 through 301 in my code in `find_lane_pipeline.py` in the function `pipeline()`.  Here is an example of my result on a test image:

Raw                        | Augmented
:-------------------------:|:-------------------------:
![alt_text][image9]        |  ![alt_text][image10]

---

### Pipeline (video)

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (wobbly lines are ok but no catastrophic failures that would cause the car to drive off the road!).

Here's a [link to my project video result](https://youtu.be/UmeZT9RTjqE). 

---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  

##### 1. The pipeline works pretty well in general, however, there is a wobble at the end of the video. This is due to imprecise identification of the lanes when there is a shadow. A fix can be introducing one of the other two channels in HLS color space such that the yellow lane and white lane are caught separately in different color space and then merged together to get a more robust result.

##### 2. Performing an average on the fitted polynomial is very important. This helps smoothen the detected drivable space in the lane. However, this is a double edged sword since calculating the average can also delay the time of correction from a failure/imprecise detection. 


##### 3. Implementing a long pipeline is very error prone and it helped to implement and check the results following the workflow described in this report. Debugging tools like ipdb was also used to fix small typos in the code.
