import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from matplotlib.pyplot import savefig
import pickle
import glob
from PIL import Image

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

def abs_sobel_thresh(img, orient='x', sobel_kernel=3, thresh=(0, 255)):
    # Calculate directional gradient
    # Apply threshold
    # Apply the following steps to img
    # 1) Convert to grayscale
    #  hls_binary = hls_select(image, thresh=(90,255))
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    # 2) Take the derivative in x or y given orient = 'x' or 'y'
    if orient == 'x':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    elif orient == 'y':
        sobel = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
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

def mag_thresh(image, sobel_kernel=3, mag_thresh=(0, 255)):
    # Calculate gradient magnitude
    # Apply threshold
    # Apply the following steps to img
    # 1) Convert to grayscale
    # hls_binary = hls_select(image, thresh=(90,255))
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    # 2) Take the gradient in x and y separately
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1)
    # 3) Calculate the magnitude
    gradmagnitude = np.sqrt(sobelx**2 + sobely**2)
    # 4) Scale to 8-bit (0 - 255) and convert to type = np.uint8
    scale_factor = np.max(gradmagnitude)/255
    gradmagnitude = (gradmagnitude/scale_factor).astype(np.uint8)
    # 5) Create a binary mask where mag thresholds are met
    mag_binary = np.zeros_like(gradmagnitude)
    mag_binary[(gradmagnitude >= mag_thresh[0]) & (gradmagnitude <= mag_thresh[1])] = 1
    return mag_binary

def dir_threshold(image, sobel_kernel=3, thresh=(0, np.pi/2)):
    # Calculate gradient direction
    # Apply threshold
    # Apply the following steps to img
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
    binary_output = np.zeros_like(s_channel)
    binary_output[(s_channel > thresh[0]) & (s_channel <= thresh[1])] = 1
    return binary_output

def plot_thresholded_images()
    # Plot the result
    f, ((ax1, ax2, ax3), (ax4, ax5, ax6)) = plt.subplots(2, 3, figsize=(20, 9))
    # f.tight_layout()
    ax1.imshow(gradx, cmap='gray')
    ax1.set_title('Thresholded Dir. Grad. x', fontsize=30)
    ax2.imshow(grady, cmap='gray')
    ax2.set_title('Thresholded Dir. Grad. y', fontsize=30)
    ax3.imshow(mag_binary, cmap='gray')
    ax3.set_title('Thresholded Grad. Mag.', fontsize=30)
    ax4.imshow(dir_binary, cmap='gray')
    ax4.set_title('Thresholded Grad. Dir.', fontsize=30)
    ax5.imshow(image)
    ax5.set_title('Original Image', fontsize=30)
    ax6.imshow(combined, cmap='gray')
    ax6.set_title('Combined Thresholded Image', fontsize=30)
    # plt.subplots_adjust(left=0., right=1, top=0.9, bottom=0.)
    savefig('thresholded_image.png')

def binary_lane(img, vertices, sobel_ksize=3, gaussian_ksize=5, gx_thresh=(0,255) \
        gy_thresh=(0,255), mag_thresh=(0,255), dir_thresh=(0, 255, hls_thresh=(0, 255))):

    image = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    # Apply each of the thresholding functions
    gradx = abs_sobel_thresh(image, orient='x', sobel_kernel=sobel_ksize, thresh=gx_thresh)
    grady = abs_sobel_thresh(image, orient='y', sobel_kernel=sobel_ksize, thresh=gy_thresh)
    mag_binary = mag_thresh(image, sobel_kernel=sobel_ksize, mag_thresh=mag_thresh)
    dir_binary = dir_threshold(image, sobel_kernel=sobel_ksize, thresh=dir_thresh)

    # Combine all of the thresholding functions
    combined = np.zeros_like(dir_binary)
    combined[((gradx == 1) & (grady == 1)) | ((mag_binary == 1) & (dir_binary == 1)) ] = 1

    s_channel = hls_select(image, thresh=hls_thresh)
    binary_output = np.zeros_like(combined)
    binary_output[(s_channel > 0) | (combined > 0)] = 1

    region_combined = region_of_interest(binary_output, vertices)

    return region_combined


ksize = 7 # Choose a larger odd number to smooth gradient measurements
kernel_size = 5

test_img = glob.glob('test_images/*')

sample_img = cv2.imread(test_img[0])
imshape = sample_img.shape
vertices = np.array([[(30,imshape[0]),(imshape[1]/2 - 10, imshape[0]/2 + 45), \
                      (imshape[1]/2 + 10, imshape[0]/2 + 45), (imshape[1] - 30,imshape[0])]], dtype=np.int32)

for idx, fname in enumerate(test_img):
    image = cv2.imread(fname)
    binary_output = binary_lane(image, vertices, ksize, kernel_size, gx_thresh=(50, 255), \
                                gy_thresh=(50, 255), mag_thresh=(60, 255), \
                                dir_thresh=(0.7, 1.10), hls_thresh=(175, 255))

    # Save image
    gray = Image.fromarray(binary_output*255)
    write_name = 'output_images/thresholded_' + fname[12:]
    write_name = write_name[:-3] + 'png'
    print(write_name)
    # cv2.imwrite(write_name,combined )
    gray.save(write_name)

