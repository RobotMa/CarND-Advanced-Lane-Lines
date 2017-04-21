import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(6,5,0)
objp = np.zeros((6*9,3), np.float32)
objp[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

# Arrays to store object points and image points from all the images.
objpoints = [] # 3d points in real world space
imgpoints = [] # 2d points in image plane.

# Make a list of calibration images
cal_images = glob.glob('camera_cal/calibration*.jpg')

# Step through the list and search for chessboard corners
for idx, fname in enumerate(cal_images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Find the chessboard corners
    ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

    # If found, add object points, image points
    if ret == True:
        print('Found object points')
        objpoints.append(objp)
        imgpoints.append(corners)

        # Draw and display the corners
        cv2.drawChessboardCorners(img, (9,6), corners, ret)
        # write_name = 'corners_found'+str(idx)+'.jpg'
        # cv2.imwrite(write_name, img)
        cv2.imshow('img', img)
        cv2.waitKey(50)

cv2.destroyAllWindows()

# Test undistortion on chessboard
sample_img = cv2.imread(cal_images[0])
img_size = (sample_img.shape[1], sample_img.shape[0])

# Do camera calibration given object points and image points
ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)

for idx, fname in enumerate(cal_images):

    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    write_name = 'output_images/undistorted_calibration' + str(idx) + '.jpg'
    cv2.imwrite(write_name,dst)

# Test undistortion on images
test_img = glob.glob('test_images/*')

for idx, fname in enumerate(test_img):
    print(fname[12:])
    img = cv2.imread(fname)
    dst = cv2.undistort(img, mtx, dist, None, mtx)

    write_name = 'output_images/undistorted_' + fname[12:]
    print(write_name)
    cv2.imwrite(write_name,dst)

# Save the camera calibration result for later use (we won't worry about rvecs / tvecs)
dist_pickle = {}
dist_pickle["mtx"] = mtx
dist_pickle["dist"] = dist
pickle.dump( dist_pickle, open( "camera_cal/wide_dist_pickle.p", "wb" ) )

