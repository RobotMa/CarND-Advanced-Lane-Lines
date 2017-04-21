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

        # number of sliding windows
        self.N_WINDOWS = 8

        # Remember x and y values of lanes in previous frame
        self.x_vec = None
        self.y_vec = None

        # Store recent x intercepts for averaging across frames
        self.bottom = deque(maxlen=10)
        self.top = deque(maxlen=10)

        # Radius of curvature
        self.radius = None

        # Store x intercepts in the previous frame
        self.last_top = None
        self.last_bottom = None

        # Store recent polynomial coefficients for averaging across frames
        self.fit0 = deque(maxlen=10)
        self.fit1 = deque(maxlen=10)
        self.fit2 = deque(maxlen=10)
        self.fitx = None
        self.pts = []

        # Count of the frames
        self.cnt = 0

    def augment_search(self, x, y):
        '''
            Search for the pixels in close proximity to the detected polynomial in the previous frame
        '''
        x_vec = []
        y_vec = []
        i = 720
        j = 630
        while j >= 0:
            yval = np.mean([i,j])
            xval = (np.mean(self.fit0))*yval**2 + (np.mean(self.fit1))*yval + (np.mean(self.fit2))
            x_idx = np.where((((xval - 25) < x)&(x < (xval + 25))&((y > j) & (y < i))))
            x_window, y_window = x[x_idx], y[x_idx]
            if np.sum(x_window) != 0:
                np.append(x_vec, x_window)
                np.append(y_vec, y_window)
            i -= 90
            j -= 90

        # Perform blind search for next search if no pixels are found
        if np.sum(x_vec) == 0:
            self.detected = False

        x_vec = np.array(x_vec).astype(np.float32)
        y_vec = np.array(y_vec).astype(np.float32)

        return x_vec, y_vec, self.detected

    def blind_search(self, x, y, image):
        '''
            Search for pixels in close promity to the detected peak in the histogram from scratch.
        '''
        midpoint = np.int(image.shape[1]/2)
        window_height = np.int(image.shape[0]/self.N_WINDOWS)

        x_vec = []
        y_vec = []

        i = 720
        j = 630

        while j >= 0:

            # Take a histogram of a window of the image
            histogram = np.sum(image[j:i,:], axis=0)

            if self == Left_Lane:
                peak = np.argmax(histogram[:midpoint])
            else:
                peak = np.argmax(histogram[midpoint:]) + 640

            x_idx = np.where((((peak - 25) < x)&(x < (peak + 25))&((y > j) & (y < i))))
            x_window, y_window = x[x_idx], y[x_idx]

            if np.sum(x_window) != 0:
                x_vec.extend(x_window)
                y_vec.extend(y_window)
            i -= 90
            j -= 90

        if np.sum(x_vec) > 0:
            self.detected = True
        else:
            y_vec = self.y_vec
            x_vec = self.x_vec

        x_vec = np.array(x_vec).astype(np.float32)
        y_vec = np.array(y_vec).astype(np.float32)

        return x_vec, y_vec, self.detected

    def fit_polynomial(self, x, y, imshape):
        """
            Fit a polynomial based on detected pixels in the
            warped, undistorted and thresholded binary image
        """

        # Calculate right polynomial fit based on detected pixels
        fit = np.polyfit(y, x, 2)

        # Calculate intercepts to extend the polynomial to the top and bottom of warped image
        bottom, top = self.get_intercepts(fit, imshape)

        # Average intercepts across 5 frames
        self.bottom.append(bottom)
        bottom = np.mean(self.bottom)
        self.top.append(top)
        top = np.mean(self.top)
        self.last_bottom = bottom
        self.last_top = top

        x = np.append(x, bottom)
        y = np.append(y, 720)
        x = np.append(x, top)
        y = np.append(y, 0)

        # Sort right lane pixels
        x, y = self.sort_vec(x, y)
        self.x_vec = x
        self.y_vec = y

        # Recalculate polynomial with intercepts and average across n frames
        fit = np.polyfit(y, x, 2)
        self.fit0.append(fit[0])
        self.fit1.append(fit[1])
        self.fit2.append(fit[2])
        fit = [np.mean(self.fit0), np.mean(self.fit1), np.mean(self.fit2)]

        # Fit polynomial to detected pixels
        fitx = fit[0]*y**2 + fit[1]*y + fit[2]
        self.fitx = fitx

        return x, y, bottom

    def sort_vec(self, x_vec, y_vec):
        sorted_index = np.argsort(y_vec)
        sorted_y_vec = y_vec[sorted_index]
        sorted_x_vec = x_vec[sorted_index]
        return sorted_x_vec, sorted_y_vec

    def get_intercepts(self, polynomial, imshape):
        bottom = polynomial[0]*imshape[0]**2 + polynomial[1]*imshape[0] + polynomial[2]
        top = polynomial[0]*0**2 + polynomial[1]*0 + polynomial[2]
        return bottom, top

    def compute_curvature(self, x_vec, y_vec):
        # Define conversion in x and y from pixels space to meters
        ym_per_pix = 30./720 # meters per pixel in y dimension
        xm_per_pix = 3.7/700 # meteres per pixel in x dimension

        # Fit new polynomials to x,y in world space
        fit_cr = np.polyfit(y_vec*ym_per_pix, x_vec*xm_per_pix, 2)

        # Calculate the new radii of curvature
        curverad = ((1 + (2*fit_cr[0]*np.max(y_vec) + fit_cr[1])**2)**1.5) \
                                     /np.absolute(2*fit_cr[0])
        return curverad

def project_back(combined_binary, undistort, Minv, Left_Lane, Right_Lane):
    """
        Draw the lanes on the warped image and project it back to the
        unwarped and undistorted road image
    """

    # create an iamge to draw the lines on
    warp_zero = np.zeros_like(combined_binary).astype(np.uint8)
    color_warp = np.dstack((warp_zero, warp_zero, warp_zero))

    # Recast the x and y points into usable format for cv2.fillPoly()
    pts_left = np.array([np.flipud(np.transpose(np.vstack([Left_Lane.fitx, Left_Lane.y_vec])))])
    pts_right = np.array([np.transpose(np.vstack([Right_Lane.fitx, Right_Lane.y_vec]))])
    pts = np.hstack((pts_left, pts_right))

    # Draw the lane onto the warped blank image
    cv2.polylines(color_warp, np.int_([pts]), isClosed=False, color=(0,0,255), thickness = 40)

    cv2.fillPoly(color_warp, np.int_(pts), (34,255,34))

    # Warp the blank back to original image space using inverse perspective matrix (Minv)
    imshape = combined_binary.shape
    newwarp = cv2.warpPerspective(color_warp, Minv, (imshape[1], imshape[0]))

    # Combine the result with the original image
    result = cv2.addWeighted(undistort, 1, newwarp, 0.5, 0)

    return result

def pipeline(image):
    """
        Pipeline for processing frames in the video
    """

    ksize = 7
    kernel_size = 5
    imshape = image.shape

    # Define vertices for masking the region of interest
    vertices = np.array([[(30,imshape[0]),(imshape[1]/2 - 10, imshape[0]/2 + 45), \
                      (imshape[1]/2 + 10, imshape[0]/2 + 45), (imshape[1] - 30,imshape[0])]], dtype=np.int32)

    # Define the area for perspective transform
    area_of_interest = [[570,460], [720,460], [1170,720], [160,720]]

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

    # import ipdb; ipdb.set_trace() #
    # Search for left lane pixels around previous polynomial
    if Left_Lane.detected == True:
        leftx, lefty, Left_Lane.detected = Left_Lane.augment_search(x, y)

    # Search for right lane pixels around previous polynomial
    if Right_Lane.detected == True:
        rightx, righty, Right_Lane.detected = Right_Lane.augment_search(x, y)

    # Search for right lane from scratch
    if Right_Lane.detected == False:
        rightx, righty, Right_Lane.detected = Right_Lane.blind_search(x, y, combined_binary)

    # Search for left lane from scratch
    if Left_Lane.detected == False:
        leftx, lefty, Left_Lane.detected = Left_Lane.blind_search(x, y, combined_binary)

    leftx, lefty, leftx_bottom = Left_Lane.fit_polynomial(leftx, lefty, imshape)
    rightx, righty, rightx_bottom = Right_Lane.fit_polynomial(rightx, righty, imshape)

    # Compute radius of curvature for each lane in meters
    left_curverad = Left_Lane.compute_curvature(leftx, lefty)
    right_curverad = Right_Lane.compute_curvature(rightx, righty)

    # Only print the radius of curvature every 3 frames for improved readability
    if Right_Lane.cnt % 3 == 0:
        Left_Lane.radius = left_curverad
        Right_Lane.radius = right_curverad

    Left_Lane.cnt += 1

    # Calculate the vehicle position relative to the center of the lane
    position = (rightx_bottom + leftx_bottom)/2
    shift_from_center = abs((imshape[1]/2 - position)*3.7/700)

    # Draw color-filled lanes on the unwarped and undistorted road image
    result = project_back(combined_binary, undistort, Minv, Left_Lane, Right_Lane)

    # Print shift from center
    if position > imshape[1]/2:
        cv2.putText(result, 'Vehicle is {:.2f}m left of center'.format(shift_from_center), (90,85),
                 fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 3)
    else:
        cv2.putText(result, 'Vehicle is {:.2f}m right of center'.format(shift_from_center), (90,85),
                 fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 3)

    # Print radius of curvature
    cv2.putText(result, 'Radius of Curvature {}(m)'.format(int((Left_Lane.radius + Right_Lane.radius)/2)),\
            (120,140),fontFace = 16, fontScale = 2, color=(255,255,255), thickness = 3)

    return result

if __name__ == "__main__":

    Left_Lane = Line()
    Right_Lane = Line()

    video_output = 'result.mp4'
    clip1 = VideoFileClip("project_video.mp4").subclip(0,2)
    white_clip = clip1.fl_image(pipeline)
    white_clip.write_videofile(video_output, audio=False)
    '''
    video_complete = 'complete.mp4'
    clip = VideoFileClip("project_video.mp4")
    white_clip = clip.fl_image(pipeline)
    whiote_clip.write_videofile(video_complete, audio=False)
    '''
