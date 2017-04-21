import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import savefig
import pickle
import glob
from PIL import Image

cnt = 0

def region_of_interest(img, vertices):
    """
    Applies an image mask.

    Only keeps the region of the image defined by the polygon
    formed from `vertices`. The rest of the image is set to black.
    """
    #defining a blank mask to start with
    mask = np.zeros_like(img)

    #defining a 3 channel or 1 channel color to fill the mask with depending on the input image
    if len(img.shape) > 2:
        channel_count = img.shape[2]  # i.e. 3 or 4 depending on your image
        ignore_mask_color = (255,) * channel_count
    else:
        ignore_mask_color = 255

    #filling pixels inside the polygon defined by "vertices" with the fill color
    cv2.fillPoly(mask, vertices, ignore_mask_color)

    #returning the image only where mask pixels are nonzero
    masked_image = cv2.bitwise_and(img, mask)
    return masked_image

def abs_sobel_threshold(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    """
        Calculate directional gradient
        Apply threshold
        Apply the following steps to img
    """
    # 1) Convert to grayscale
    #  hls_binary = hls_select(image, thresh=(90,255))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the derivative or gradient
    abs_sobel = np.absolute(sobel)
    # 4) Scale to 8-bit (0 - 255) then convert to type = np.uint8
    scaled_sobel = np.uint8(255*abs_sobel/np.max(abs_sobel))
    # 5) Create a mask of 1's where the scaled gradient magnitude
    # is > thresh_min and < thresh_max
    grad_binary = np.zeros_like(scaled_sobel)
    grad_binary[(scaled_sobel >= thresh[0]) & (scaled_sobel <= thresh[1])] = 1
    # 6) Return this mask as your binary_output image
    return grad_binary

def mag_threshold(image, sobel_kernel=3, thresh=(0, 255)):
    """
        Calculate gradient magnitude
        Apply threshold
        Apply the following steps to img
    """
    # 1) Convert to grayscale
    # hls_binary = hls_select(image, thresh=(90,255))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Calculate the magnitude
    gradmagnitude = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmagnitude)/255
    gradmagnitude = (gradmagnitude/scale_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(gradmagnitude)
    mag_binary[(gradmagnitude >= thresh[0]) & (gradmagnitude <= thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    """
        Calculate gradient direction
        Apply threshold
        Apply the following steps to img
    """
    # 1) Convert to the S Channel of HLS color space
    # hls_binary = hls_select(image, thresh=(90,255))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize = sobel_kernel)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize = sobel_kernel)
    # 3) Take the absolute value of the x and y gradients
    abs_sobelx = np.abs(sobelx)
    abs_sobely = np.abs(sobely)
    # 4) Use np.arctan2(abs_sobely, abs_sobelx) to calculate the direction of the gradient
    dir_grad = np.arctan2(abs_sobely, abs_sobelx)
    # 5) Create a binary mask where direction thresholds are met
    dir_binary = np.zeros_like(gray)
    dir_binary[(dir_grad >= thresh[0]) & (dir_grad <= thresh[1])] = 1
    return dir_binary

def hls_select(img, thresh=(0,255)):
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
    s_channel = hls[:,:,2]
    binary_hls = np.zeros_like(s_channel)
    binary_hls[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_hls

def plot_thresholded_images(img, gradx, grady, mag_binary, dir_binary, combined):
    # Plot the result
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 9))
    ax1.imshow(gradx, cmap='gray')
    ax1.set_title('Thresholded Dir. Grad. x', fontsize=30)
    ax2.imshow(grady, cmap='gray')
    ax2.set_title('Thresholded Dir. Grad. y', fontsize=30)
    ax3.imshow(mag_binary, cmap='gray')
    ax3.set_title('Thresholded Grad. Mag.', fontsize=30)
    ax4.imshow(dir_binary, cmap='gray')
    ax4.set_title('Thresholded Grad. Dir.', fontsize=30)
    ax5.imshow(img)
    ax5.set_title('Original Image', fontsize=30)
    ax6.imshow(combined, cmap='gray')
    ax6.set_title('Combined Thresholded Image', fontsize=30)
    savefig('thresholded_image.png')

def binary_lane(img, vertices, sobel_ksize=3, gaussian_ksize=5, gx_thresh=(0,255), \
        gy_thresh=(0,255), mag_thresh=(0,255), dir_thresh=(0, 5), hls_thresh=(0, 255),  plot_opt=False):

    image = cv2.GaussianBlur(img, (gaussian_ksize, gaussian_ksize), 0)

    # Apply each of the thresholding functions
    gradx = abs_sobel_threshold(image, orient='x', sobel_kernel=sobel_ksize, thresh=gx_thresh)
    grady = abs_sobel_threshold(image, orient='y', sobel_kernel=sobel_ksize, thresh=gy_thresh)
    mag_binary = mag_threshold(image, sobel_kernel=sobel_ksize, thresh=mag_thresh)
    dir_binary = dir_threshold(image, sobel_kernel=sobel_ksize, thresh=dir_thresh)

    # Combine all of the thresholding functions
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) ] = 1

    s_channel = hls_select(image, thresh=hls_thresh)
    binary_out = np.zeros_like(combined)
    binary_out[(s_channel > 0) | (combined > 0)] = 1

    region_combined = region_of_interest(binary_out, vertices)

    if plot_opt == True:
        # Plot the masked raw image
        plot_region(img, vertices)
        # Plot the thresholded image
        plot_thresholded_images(img, gradx, grady, mag_binary, dir_binary, region_combined)

    return region_combined

def plot_region(img, vertices):

    global cnt

    # convert numpy array to tuples for cv2.line
    point = []
    if type(vertices) is np.ndarray:
        point.append((vertices[0,0,0], vertices[0,0,1]))
        point.append((vertices[0,1,0], vertices[0,1,1]))
        point.append((vertices[0,2,0], vertices[0,2,1]))
        point.append((vertices[0,3,0], vertices[0,3,1]))
    elif type(vertices) is list:
        point.append((vertices[0][0], vertices[0][1]))
        point.append((vertices[1][0], vertices[1][1]))
        point.append((vertices[2][0], vertices[2][1]))
        point.append((vertices[3][0], vertices[3][1]))


    img_region = cv2.line(img, point[0], point[1], [255, 0, 0], thickness=1)
    img_region = cv2.line(img_region, point[1], point[2], [255, 0, 0], thickness=1)
    img_region = cv2.line(img_region, point[2], point[3], [255, 0, 0], thickness=1)
    img_region = cv2.line(img_region, point[3], point[0], [255, 0, 0], thickness=1)

    img_region = Image.fromarray(img_region)
    write_name = 'output_images/' + str(cnt) + '.png'
    img_region.save(write_name)
    cnt += 1

def perspective_transform(img, src, mtx, dist):

    undistort = cv2.undistort(img, mtx, dist, None, mtx)

    # Choose offset from image corners to plot detected corners
    # This should be chosen to present the result at the proper aspect ratio
    # My choice of 100 pixels is not exact, but close enough for our purpose here
    offset = 100 # offset for dst points
    offset1 = 0
    # Grab the image shape
    img_size = (img.shape[1], img.shape[0])

    # plot_region(img, src)

    # For source points I'm grabbing the outer four detected corners
    src = np.float32(src)

    # For destination points, I'm arbitrarily choosing some points to be
    # a nice fit for displaying our warped result
    dst = np.float32([[offset, offset1],
                      [img_size[0]-offset, offset1],
                      [img_size[0]-offset, img_size[1]],
                      [offset, img_size[1]]])


    # Given src and dst points, calculate the perspective transform matrix
    M = cv2.getPerspectiveTransform(src, dst)

    # Given dst and src points, calculate the inverse of perspective transform matrix
    Minv = cv2.getPerspectiveTransform(dst, src)

    # Warp the image using OpenCV warpPerspective()
    warped = cv2.warpPerspective(undistort, M, img_size)

    # Return the resulting image and matrix
    return warped, M, Minv

if __name__ == "__main__":

    ksize = 7 # Choose a larger odd number to smooth gradient measurements
    kernel_size = 5

    test_img = glob.glob('test_images/*')
    sample_img = cv2.imread(test_img[0])

    imshape = sample_img.shape
    vertices = np.array([[(30,imshape[0]),(imshape[1]/2 - 10, imshape[0]/2 + 45), \
                      (imshape[1]/2 + 10, imshape[0]/2 + 45), (imshape[1] - 30,imshape[0])]], dtype=np.int32)

    area_of_interest = [[150+430-10,460],[1150-440 + 10,460],[1140 + 30,720],[180-20,720]]
    # area_of_interest = [[150+430,460],[1150-440,460],[1150,720],[150,720]]

    # Load the calibrated parameters dist and mtx stored in pickle file
    dist_pickle = pickle.load( open("camera_cal/wide_dist_pickle.p", "rb"))

    for idx, fname in enumerate(test_img):
        image = cv2.imread(fname)
        binary_output = binary_lane(image, vertices, ksize, kernel_size, gx_thresh=(50, 255), \
                                gy_thresh=(50, 255), mag_thresh=(60, 255), dir_thresh=(0.7, 1.10), hls_thresh=(160, 255), plot_opt=True)

        warped, M, Minv = perspective_transform(binary_output, area_of_interest, dist_pickle['mtx'], dist_pickle['dist'])


        # Save image
        gray = Image.fromarray(binary_output*255)
        write_name = 'output_images/thresholded_' + fname[12:]
        write_name = write_name[:-3] + 'png'
        gray.save(write_name)

        warped_image = Image.fromarray(warped*255)
        write_name_warped = 'output_images/warped_' + fname[12:]
        write_name_wapred = write_name[:-3] + 'png'
        warped_image.save(write_name_warped)

