import numpy as np
from collections import deque
from moviepy.editor import VideoFileClip
import cv2
import glob
from find_lane import *

class Line:
    def __init__(self):
        # Was the line found in the previous frame?
        self.detected = False

        # Remember x and y values of lanes in previous frame
        self.X = None
        self.Y = None

        # Store recent x intercepts for averaging across frames
        self.x_int = deque(maxlen=10)
        self.top = deque(maxlen=10)

        # Remember previous x intercept to compare against current one
        self.lastx_int = None
        self.last_top = None

        # Remember radius of curvature
        self.radius = None

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = deque(maxlen=10)
        self.fit1 = deque(maxlen=10)
        self.fit2 = deque(maxlen=10)
        self.fitx = None
        self.pts = []

        # Count the number of frames
        self.count = 0

    def found_search(self, x, y):
        '''
        This function is applied when the lane lines have been detected in the previous frame.
        It uses a sliding window to search for lane pixels in close proximity (+/- 25 pixels in the x direction)
        around the previous detected polynomial.
        '''
        xvals = []
        yvals = []
        if self.detected == True:
            i = 720
            j = 630
            while j >= 0:
                yval = np.mean([i,j])
                xval = (np.mean(self.fit0))*yval**2 + (np.mean(self.fit1))*yval + (np.mean(self.fit2))
                x_idx = np.where((((xval - 25) < x)&(x < (xval + 25))&((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    np.append(xvals, x_window)
                    np.append(yvals, y_window)
                i -= 90
                j -= 90
        if np.sum(xvals) == 0:
            self.detected = False # If no lane pixels were detected then perform blind search
        return xvals, yvals, self.detected

    def blind_search(self, x, y, image):
        '''
        This function is applied in the first few frames and/or if the lane was not successfully detected
        in the previous frame. It uses a slinding window approach to detect peaks in a histogram of the
        binary thresholded image. Pixels in close proimity to the detected peaks are considered to belong
        to the lane lines.
        '''
        xvals = []
        yvals = []
        if self.detected == False:
            i = 720
            j = 630
            while j >= 0:
                histogram = np.sum(image[j:i,:], axis=0)
                if self == Right_Lane:
                    peak = np.argmax(histogram[640:]) + 640
                else:
                    peak = np.argmax(histogram[:640])
                x_idx = np.where((((peak - 25) < x)&(x < (peak + 25))&((y > j) & (y < i))))
                x_window, y_window = x[x_idx], y[x_idx]
                if np.sum(x_window) != 0:
                    xvals.extend(x_window)
                    yvals.extend(y_window)
                i -= 90
                j -= 90

        if np.sum(xvals) > 0:
            self.detected = True
        else:
            yvals = self.Y
            xvals = self.X

        return xvals, yvals, self.detected

    def radius_of_curvature(self, xvals, yvals):
        # Define conversion in x and y from pixels space to meters
        ym_per_pix = 30./720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(yvals*ym_per_pix, xvals*xm_per_pix, 2)

        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*np.max(yvals) + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        return curverad

    def sort_vals(self, xvals, yvals):
        sorted_index = np.argsort(yvals)
        sorted_yvals = yvals[sorted_index]
        sorted_xvals = xvals[sorted_index]
        return sorted_xvals, sorted_yvals

    def get_intercepts(self, polynomial):
        bottom = polynomial[0]*720**2 + polynomial[1]*720 + polynomial[2]
        top = polynomial[0]*0**2 + polynomial[1]*0 + polynomial[2]
        return bottom, top

# Video Processing Pipeline
def process_vid(image):

    ksize = 7
    kernel_size = 5
    imshape = image.shape

    # Define vertices for masking the region of interest
    vertices = np.array([[(30,imshape[0]),(imshape[1]/2 - 10, imshape[0]/2 + 45), \
                      (imshape[1]/2 + 10, imshape[0]/2 + 45), (imshape[1] - 30,imshape[0])]], dtype=np.int32)

    # Define the area for perspective transform
    area_of_interest = [[150+420,460], [1150-430,460], [1170,720], [160,720]]

    # Load the calibrated parameters dist and mtx stored in pickle file
    dist_pickle = pickle.load( open("camera_cal/wide_dist_pickle.p", "rb"))

    # Undistort the raw image
    undistort = cv2.undistort(image, dist_pickle['mtx'], dist_pickle['dist'], None, dist_pickle['mtx'])

    # Compute the binary image using various thresholding techniques
    binary_output = binary_lane(undistort, vertices, ksize, kernel_size, gx_thresh=(50, 255), \
                                gy_thresh=(50, 255), mag_thresh=(60, 255), dir_thresh=(0.7, 1.10), hls_thresh=(160, 255))

    # Perform perspective transform
    combined_binary, M, Minv = perspective_transform(binary_output, area_of_interest, dist_pickle['mtx'], dist_pickle['dist'])


    # Identify all non zero pixels in the image
    x, y = np.nonzero(np.transpose(combined_binary))

    # Search for left lane pixels around previous polynomial
    if Left_Lane.detected == True:
        leftx, lefty, Left_Lane.detected = Left_Lane.found_search(x, y)

    # Search for right lane pixels around previous polynomial
    if Right_Lane.detected == True:
        rightx, righty, Right_Lane.detected = Right_Lane.found_search(x, y)

    # Perform blind search for right lane lines
    if Right_Lane.detected == False:
        rightx, righty, Right_Lane.detected = Right_Lane.blind_search(x, y, combined_binary)

    # Perform blind search for left lane lines
    if Left_Lane.detected == False:
        leftx, lefty, Left_Lane.detected = Left_Lane.blind_search(x, y, combined_binary)

    lefty = np.array(lefty).astype(np.float32)
    leftx = np.array(leftx).astype(np.float32)
    righty = np.array(righty).astype(np.float32)
    rightx = np.array(rightx).astype(np.float32)

    # Calculate left polynomial fit based on detected pixels
    left_fit = np.polyfit(lefty, leftx, 2)

    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    leftx_int, left_top = Left_Lane.get_intercepts(left_fit)

    # Average intercepts across n frames
    Left_Lane.x_int.append(leftx_int)
    Left_Lane.top.append(left_top)
    leftx_int = np.mean(Left_Lane.x_int)
    left_top = np.mean(Left_Lane.top)
    Left_Lane.lastx_int = leftx_int
    Left_Lane.last_top = left_top

    # Add averaged intercepts to current x and y vals
    leftx = np.append(leftx, leftx_int)
    lefty = np.append(lefty, 720)
    leftx = np.append(leftx, left_top)
    lefty = np.append(lefty, 0)

    # Sort detected pixels based on the yvals
    leftx, lefty = Left_Lane.sort_vals(leftx, lefty)

    Left_Lane.X = leftx
    Left_Lane.Y = lefty

    # Recalculate polynomial with intercepts and average across n frames
    left_fit = np.polyfit(lefty, leftx, 2)
    Left_Lane.fit0.append(left_fit[0])
    Left_Lane.fit1.append(left_fit[1])
    Left_Lane.fit2.append(left_fit[2])
    left_fit = [np.mean(Left_Lane.fit0),
                np.mean(Left_Lane.fit1),
                np.mean(Left_Lane.fit2)]

    # Fit polynomial to detected pixels
    left_fitx = left_fit[0]*lefty**2 + left_fit[1]*lefty + left_fit[2]
    Left_Lane.fitx = left_fitx

    # Calculate right polynomial fit based on detected pixels
    right_fit = np.polyfit(righty, rightx, 2)

    # Calculate intercepts to extend the polynomial to the top and bottom of warped image
    rightx_int, right_top = Right_Lane.get_intercepts(right_fit)

    # Average intercepts across 5 frames
    Right_Lane.x_int.append(rightx_int)
    rightx_int = np.mean(Right_Lane.x_int)
    Right_Lane.top.append(right_top)
    right_top = np.mean(Right_Lane.top)
    Right_Lane.lastx_int = rightx_int
    Right_Lane.last_top = right_top
    rightx = np.append(rightx, rightx_int)
    righty = np.append(righty, 720)
    rightx = np.append(rightx, right_top)
    righty = np.append(righty, 0)

    # Sort right lane pixels
    rightx, righty = Right_Lane.sort_vals(rightx, righty)
    Right_Lane.X = rightx
    Right_Lane.Y = righty

    # Recalculate polynomial with intercepts and average across n frames
    right_fit = np.polyfit(righty, rightx, 2)
    Right_Lane.fit0.append(right_fit[0])
    Right_Lane.fit1.append(right_fit[1])
    Right_Lane.fit2.append(right_fit[2])
    right_fit = [np.mean(Right_Lane.fit0), np.mean(Right_Lane.fit1), np.mean(Right_Lane.fit2)]

    # Fit polynomial to detected pixels
    right_fitx = right_fit[0]*righty**2 + right_fit[1]*righty + right_fit[2]
    Right_Lane.fitx = right_fitx

    # Compute radius of curvature for each lane in meters
    left_curverad = Left_Lane.radius_of_curvature(leftx, lefty)
    right_curverad = Right_Lane.radius_of_curvature(rightx, righty)

    # Only print the radius of curvature every 3 frames for improved readability
    if Left_Lane.count % 3 == 0:
        Left_Lane.radius = left_curverad
        Right_Lane.radius = right_curverad

    # Calculate the vehicle position relative to the center of the lane
    position = (rightx_int+leftx_int)/2
    distance_from_center = abs((640 - position)*3.7/700)

    # create an iamge to draw the lines on
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.flipud(np.transpose(np.vstack([Left_Lane.fitx, Left_Lane.Y])))])
    pts_right = np.array([np.transpose(np.vstack([right_fitx, Right_Lane.Y]))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)

        cv2.fillPoly(color_warp, np.int_(pts), (34,255,34))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    newwarp = cv2.warpPerspective(color_warp, Minv, (image.shape[1], image.shape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.5, 0)

    # Print distance from center on video
    if position > 640:
        cv2.putText(result, 'Vehicle is {:.2f}m left of center'.format(distance_from_center), (100,80),
                 fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 3)
    else:
        cv2.putText(result, 'Vehicle is {:.2f}m right of center'.format(distance_from_center), (100,80),
                 fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 3)

    # Print radius of curvature on video
    cv2.putText(result, 'Radius of Curvature {}(m)'.format(int((Left_Lane.radius+Right_Lane.radius)/2)), (120,140),
             fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 3)
    Left_Lane.count += 1
    return result


Left_Lane = Line()
Right_Lane = Line()
video_output = 'result.mp4'
video_complete = 'complete.mp4'
clip1 = VideoFileClip("project_video.mp4").subclip(0,2)
white_clip = clip1.fl_image(process_vid)
white_clip.write_videofile(video_output, audio=False)
clip = VideoFileClip("project_video.mp4")
white_clip = clip.fl_image(process_vid)
white_clip.write_videofile(video_complete, audio=False)
