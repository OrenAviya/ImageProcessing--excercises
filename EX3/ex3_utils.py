import sys
from typing import List

import numpy as np
import cv2
from numpy.linalg import LinAlgError
import matplotlib.pyplot as plt


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 322273301

# ---------------------------------------------------------------------------
# ------------------------ Lucas Kanade optical flow ------------------------
# ---------------------------------------------------------------------------

def compute_gradients(image):
    """
    Computes the gradients Ix and Iy for an image
    :param image: Input image
    :return: Gradients Ix and Iy
    """
    # Convert the image to grayscale if it's not already
    if len(image.shape) > 2:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    else:
        gray = image

    # Calculate the gradients Ix and Iy using Sobel operator
    ix = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    iy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)

    return ix, iy
"""
The cv2.Sobel() function is then applied to the grayscale image to compute the gradients. It takes the following parameters:

gray: The grayscale image
cv2.CV_64F: The data type of the output gradients (64-bit floating-point)
1: Order of the derivative in the x direction (Ix)
0: Order of the derivative in the y direction (Iy)
ksize=3: Size of the Sobel kernel (3x3)
The function returns the computed gradients Ix and Iy.

Once you have the gradients Ix and Iy, you can use them to calculate the optical flow using the Lucas-Kanade algorithm.

"""
def calc_least_squere(I_x , I_y , I_t , mask_slice):
    I_x_slice , I_y_slice , I_t_slice = I_x[mask_slice] , I_y[mask_slice] , I_t[mask_slice] 
    At_mult_A = [[(I_x_slice * I_x_slice).sum() , (I_x_slice * I_y_slice).sum()],
                 [(I_x_slice * I_y_slice).sum() , (I_y_slice * I_y_slice).sum()]]
    b = [[-(I_x_slice * I_t_slice).sum()] , [-(I_y_slice * I_t_slice).sum()]]
    u_v = np.linalg.inv(At_mult_A) @ b
    return u_v

def lucas_kanade_optical_flow(Ix, Iy, It, win_size=5):
    """
    Computes the optical flow using the Lucas-Kanade algorithm
    :param Ix: Gradient along x-axis
    :param Iy: Gradient along y-axis
    :param It: Image difference (Im1 - Im2)
    :param win_size: Optical flow window size (odd number)
    :return: Estimated motion parameters (u, v)
    """
    height, width = Ix.shape
    half_win = win_size // 2
    u = np.zeros_like(Ix)
    v = np.zeros_like(Iy)

    for i in range(half_win, height - half_win):
        for j in range(half_win, width - half_win):
            I_x_slice = Ix[i - half_win: i + half_win + 1, j - half_win: j + half_win + 1].flatten()
            I_y_slice = Iy[i - half_win: i + half_win + 1, j - half_win: j + half_win + 1].flatten()
            I_t_slice = It[i - half_win: i + half_win + 1, j - half_win: j + half_win + 1].flatten()
            
            mask_slice = (I_x_slice != 0) & (I_y_slice != 0)
            if mask_slice.any():
                u_v = calc_least_squere(I_x_slice, I_y_slice, I_t_slice, mask_slice)
                u[i, j] = u_v[0]
                v[i, j] = u_v[1]
    return u, v

def opticalFlow(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> tuple[np.ndarray, np.ndarray]:
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    if len(im1.shape) > 2:
        gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = im1.copy()

    if len(im2.shape) > 2:
        gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = im2.copy()

 # if the images don't have the same shape we will throw an error
    if im1.shape != im2.shape:
        raise Exception ("The images must be in the same size")

    # Convert images to float32
    gray1 = im1.astype(np.float32)
    gray2 = im2.astype(np.float32)

    # Compute image gradients
    I_x, I_y = compute_gradients(gray1)

    # Compute image difference It
    difference = gray1 - gray2
    I_t = difference.astype(np.float32)

    # Get image shape
    height, width = gray1.shape

    # Calculate optical flow
    original_points = []
    displacement_vectors = []

    half_win = win_size // 2

    for i in range(half_win, height - half_win, step_size):
        for j in range(half_win, width - half_win, step_size):
            # Extract gradient slices
            I_x_slice = I_x[i - half_win: i + half_win + 1, j - half_win: j + half_win + 1].flatten()
            I_y_slice = I_y[i - half_win: i + half_win + 1, j - half_win: j + half_win + 1].flatten()
            I_t_slice = I_t[i - half_win: i + half_win + 1, j - half_win: j + half_win + 1].flatten()

            # Mask for valid points
            mask_slice = np.isfinite(I_x_slice) & np.isfinite(I_y_slice) & np.isfinite(I_t_slice)

            if mask_slice.sum() == win_size * win_size:
                # Calculate displacement vector using least squares
                u_v = calc_least_squere(I_x_slice, I_y_slice, I_t_slice, mask_slice)
                original_points.append([j, i])
                displacement_vectors.append(u_v.flatten())

    return np.array(original_points), np.array(displacement_vectors)

# def opticalFlow2(im1: np.ndarray, im2: np.ndarray, step_size=10, win_size=5) -> tuple[np.ndarray, np.ndarray]:
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each point
    """

    # Convert the input images to grayscale if they are not already
    if len(im1.shape) > 2:
        gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
    else:
        gray1 = im1.copy()

    if len(im2.shape) > 2:
        gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)
    else:
        gray2 = im2.copy()

    # Compute the gradients Ix and Iy
    Ix, Iy = compute_gradients(gray1)

    # Initialize the output arrays for original points and motion vectors
    points = []
    motion_vectors = []

    # Iterate over the image pixels with the specified step size
    for i in range(win_size // 2, gray1.shape[0] - win_size // 2, step_size):
        for j in range(win_size // 2, gray1.shape[1] - win_size // 2, step_size):
            # Extract slices of Ix, Iy, and It around the current pixel
            I_x_slice = Ix[i - win_size // 2: i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1].flatten()
            I_y_slice = Iy[i - win_size // 2: i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1].flatten()
            I_t_slice = gray1[i - win_size // 2: i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1].astype(np.float32) - gray2[i - win_size // 2: i + win_size // 2 + 1, j - win_size // 2: j + win_size // 2 + 1].astype(np.float32)

            # Calculate the motion vector using the least squares method
            u_v = calc_least_squere(I_x_slice, I_y_slice, I_t_slice, win_size)

            # Append the original point and motion vector to the output arrays
            points.append([j, i])
            motion_vectors.append(u_v)

    # Convert the output arrays to NumPy arrays
    points = np.array(points)
    motion_vectors = np.array(motion_vectors)

    return points, motion_vectors

# def opticalFlow3(im1: np.ndarray, im2: np.ndarray, step_size=10,win_size=5) -> tuple[np.ndarray, np.ndarray]:
    """
    Given two images, returns the Translation from im1 to im2
    :param im1: Image 1
    :param im2: Image 2
    :param step_size: The image sample size
    :param win_size: The optical flow window size (odd number)
    :return: Original points [[x,y]...], [[dU,dV]...] for each points
    """
    
    # Convert images to grayscale
    if ((len(im1.shape)<3) and  len(im2.shape)<3 ):
        gray1 = im1
        gray2 = im2
    else:    
        gray1 = cv2.cvtColor(im1, cv2.COLOR_BGR2GRAY)
        gray2 = cv2.cvtColor(im2, cv2.COLOR_BGR2GRAY)

    Ix , Iy = compute_gradients(gray1)
    difference =  gray1.astype(np.float32) - gray2.astype(np.float32) #It = Im1-Im2
    '''
    we want to minimize the squere of sigma on all pixels (u,v)
    :(It + Ix*u +Iy*v)^2
    '''
    # Define parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(win_size, win_size),
                     maxLevel=2,
                     criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Create an array to store the original points and the displacement vectors
    original_points = []
    displacement_vectors = []

    # Calculate optical flow for each pixel in the image
    for y in range(0, im1.shape[0], step_size):
        for x in range(0, im1.shape[1], step_size):
            # Define the search window
            p0 = np.array([[x, y]], dtype=np.float32)
            p1, st, err = cv2.calcOpticalFlowPyrLK(gray1, gray2, p0, None, **lk_params)

            # Calculate the displacement vector
            dx = p1[0, 0] - p0[0, 0]
            dy = p1[0, 1] - p0[0, 1]

            # Append the original point and the displacement vector to the respective arrays
            original_points.append([x, y])
            displacement_vectors.append([dx, dy])

    # Convert the arrays to NumPy arrays
    original_points = np.array(original_points)
    displacement_vectors = np.array(displacement_vectors)

    return original_points, displacement_vectors



def opticalFlowPyrLK(img1: np.ndarray, img2: np.ndarray, k: int, stepSize: int, winSize: int) -> np.ndarray:
    """
    :param img1: First image
    :param img2: Second image
    :param k: Pyramid depth
    :param stepSize: The image sample size
    :param winSize: The optical flow window size (odd number)
    :return: A 3d array, with a shape of (m, n, 2),
    where the first channel holds U, and the second V.
    """
    # ********
    
    uv_return = []
    xy_return = []
    img1_pyr = gaussianPyr(img1, k)
    img2_pyr = gaussianPyr(img2, k)
    # entering the last pyramid
    x_y_prev, u_v_prev = opticalFlow(img1_pyr[-1], img2_pyr[-1], stepSize, winSize)
    x_y_prev = list(x_y_prev)
    u_v_prev = list(u_v_prev)
    for i in range(1, k):
        # find optical flow for this level
        x_y_i, uv_i = opticalFlow(img1_pyr[-1 - i], img2_pyr[-1 - i], stepSize, winSize)
        uv_i = list(uv_i)
        x_y_i = list(x_y_i)
        for g in range(len(x_y_i)):
            x_y_i[g] = list(x_y_i[g])
            # uv_i[g] = list(uv_i[g])
        # update uv according to formula
        for j in range(len(x_y_prev)):
            x_y_prev[j] = [element * 2 for element in x_y_prev[j]]
            u_v_prev[j] = [element * 2 for element in u_v_prev[j]]
        # If location of movements we found are new then append them, else add them to the proper location
        for j in range(len(x_y_i)):
            if x_y_i[j] in x_y_prev:
                u_v_prev[j] += uv_i[j]
            else:
                x_y_prev.append(x_y_i[j])
                u_v_prev.append(uv_i[j])
    # now we shall change uv and xy to a 3 dimensional array
    arr3d = np.zeros(shape=(img1.shape[0], img1.shape[1], 2))
    for x in range(img1.shape[0]):
        for y in range(img1.shape[1]):
            if [y, x] not in x_y_prev:
                arr3d[x, y] = [0, 0]
            else:
                arr3d[x, y] = u_v_prev[x_y_prev.index([y, x])]
    return arr3d
    # ********
     # Convert images to grayscale
    if len(img1.shape) > 2:
        img1_gray = cv2.cvtColor(img1, cv2.COLOR_BGR2GRAY)
    else:
        img1_gray = img1.copy()

    if len(img2.shape) > 2:
        img2_gray = cv2.cvtColor(img2, cv2.COLOR_BGR2GRAY)
    else:
        img2_gray = img2.copy()
   
    # if the images don't have the same shape we will throw an error
    if img1.shape != img2.shape:
        raise Exception ("The images must be in the same size")

    # Create image pyramids
    pyramid1 = [img1_gray]
    pyramid2 = [img2_gray]

    for i in range(1, k):
        img1_gray = cv2.pyrDown(img1_gray)
        img2_gray = cv2.pyrDown(img2_gray)
        pyramid1.append(img1_gray)
        pyramid2.append(img2_gray)

    # Initialize optical flow with zeros
    optical_flow = np.zeros_like(pyramid1[0], dtype=np.float32)

    # Iterate through the pyramid levels
    for level in range(k - 1, -1, -1):
        scaled_optical_flow = cv2.pyrUp(optical_flow)

        # Calculate optical flow for the current level
        u, v = opticalFlow(pyramid1[level], pyramid2[level], stepSize, winSize)

     # Reshape u and v to match the shape of scaled_optical_flow
        # u_resized = cv2.resize(u, scaled_optical_flow.shape[::-1])
        # v_resized = cv2.resize(v, scaled_optical_flow.shape[::-1])

        # Resize u and v to match the shape of scaled_optical_flow
        u_resized = cv2.resize(u, (scaled_optical_flow.shape[1], scaled_optical_flow.shape[0]), interpolation=cv2.INTER_LINEAR)
        v_resized = cv2.resize(v, (scaled_optical_flow.shape[1], scaled_optical_flow.shape[0]), interpolation=cv2.INTER_LINEAR)

        # Update optical flow with current level's flow
        # optical_flow = scaled_optical_flow + np.stack([u, v], axis=2) * 2.0
        
        # Update optical flow with current level's flow
        optical_flow = scaled_optical_flow + np.stack([u_resized, v_resized], axis=2)

    return optical_flow

# ---------------------------------------------------------------------------
# ------------------------ Image Alignment & Warping ------------------------
# ---------------------------------------------------------------------------
def findTranslationLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 2 in grayscale format.
    :return: Translation matrix by LK.
    """
    # Set the parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Find feature points in the first image
    features1 = cv2.goodFeaturesToTrack(im1, mask=None, maxCorners=100, qualityLevel=0.01, minDistance=10)

    if features1 is None or len(features1) == 0:
        raise ValueError("No valid feature points found in the first image.")

    # Ensure im1 and im2 have the same dimensions
    if im1.shape != im2.shape:
        im2 = cv2.resize(im2, (im1.shape[1], im1.shape[0]))

    try:
        # Calculate optical flow between the first and second image
        features2, status, _ = cv2.calcOpticalFlowPyrLK(im1, im2, features1, None, **lk_params)
    except cv2.error as e:
        raise ValueError("Error occurred during optical flow calculation: " + str(e))

    # Filter out the good feature points that were successfully tracked
    good1 = features1[status == 1]
    good2 = features2[status == 1]

    if len(good1) == 0 or len(good2) == 0:
        raise ValueError("No valid feature points found in the second image.")

    # Calculate the mean displacement in x and y directions
    dx = np.mean(good2[:, 0] - good1[:, 0])
    dy = np.mean(good2[:, 1] - good1[:, 1])
# Create the translation matrix
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    # Return the translation matrix
    return translation_matrix


def findTranslationLK0(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by LK.
    """
    '''
    in that function the assumption is that the transformation between the two images is a simple translation. 
    The cv2.calcOpticalFlowPyrLK() function calculates the optical flow between im1 and im2 using the Lucas-Kanade method.
    It returns the tracked feature points in features2,
    the status of each feature point in status, 
    and the errors between the tracked points and the actual points in _.
    
    '''
      # Set the parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Find feature points in the first image
    features1 = cv2.goodFeaturesToTrack(im1, mask=None, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # Calculate optical flow between the first and second image
    features2, status, _ = cv2.calcOpticalFlowPyrLK(im1, im2, features1, None, **lk_params)

    # Filter out the good feature points that were successfully tracked
    good1 = features1[status == 1]
    good2 = features2[status == 1]

    # Calculate the mean displacement in x and y directions
    dx = np.mean(good2[:, 0, 0] - good1[:, 0, 0])
    dy = np.mean(good2[:, 0, 1] - good1[:, 0, 1])

 # Create the translation matrix
    translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])

    # Return the translation matrix
    return translation_matrix


def findRigidLK(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by LK.
    """
     # Set the parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(15, 15), maxLevel=2, criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    # Find feature points in the first image
    features1 = cv2.goodFeaturesToTrack(im1, mask=None, maxCorners=100, qualityLevel=0.01, minDistance=10)

    # Calculate optical flow between the first and second image
    features2, status, _ = cv2.calcOpticalFlowPyrLK(im1, im2, features1, None, **lk_params)

    # Filter out the good feature points that were successfully tracked
    good1 = features1[status == 1]
    good2 = features2[status == 1]

    # Estimate translation vector
    dx = np.mean(good2[:, 0] - good1[:, 0])
    dy = np.mean(good2[:,1] - good1[:, 1])

    # Estimate rotation angle
    dtheta = np.mean(np.arctan2(good2[:, 1] - good1[:, 1], good2[:, 0] - good1[:, 0]))

    # Return the rigid transformation matrix [dx, dy, angle]
    return np.array([dx, dy, dtheta])

import cv2
import numpy as np

def estimate_translation_matrix(image1, image2):
    # Convert the images to numpy arrays
    image1 = np.array(image1, dtype=np.float32)
    image2 = np.array(image2, dtype=np.float32)
    
    # trying to improve the model with cleaning nnoises:
    # denoised_image1 = cv2.GaussianBlur(image1, (3, 3), 0)
    # denoised_image2 = cv2.GaussianBlur(image2, (3, 3), 0)
    denoised_image1 = cv2.medianBlur(image1, 3)
    denoised_image2 = cv2.medianBlur(image2, 3)
    
    # Normalize the images to have zero mean and unit variance
    image1_normalized = cv2.normalize(denoised_image1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image2_normalized = cv2.normalize(denoised_image2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Calculate the cross-correlation
    cross_correlation = cv2.matchTemplate(image1_normalized, image2_normalized, cv2.TM_CCORR_NORMED)
    
    # Find the position of the maximum correlation
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(cross_correlation)
    print (f'minval: {min_val} , maxval: {max_val} , minloc: {min_loc} , maxloc: {max_loc}')
    translation_x = max_loc[0]
    translation_y = max_loc[1]
    
    # Create the translation matrix
    translation_matrix = np.float32([[1, 0, translation_x], [0, 1, translation_y]])

    return translation_matrix


def findTranslationCorr0(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
     # Convert the images to numpy arrays
    im1 = np.array(im1, dtype=np.float32)
    im2= np.array(im2, dtype=np.float32)

    # Normalize the images to have zero mean and unit variance
    image1_normalized = cv2.normalize(im1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    image2_normalized = cv2.normalize(im2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

    # Calculate the cross-correlation
    cross_correlation = cv2.matchTemplate(image1_normalized, image2_normalized, cv2.TM_CCORR_NORMED)
    correlation = np.max(cross_correlation)
    return correlation
 
     # Normalize the images
    norm_im1 = cv2.normalize(im1, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    norm_im2 = cv2.normalize(im2, None, alpha=0, beta=1, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)
    # Preprocess the images if necessary

    # Build image pyramid for both images
    pyramid_im1 = [im1]
    pyramid_im2 = [im2]
    for _ in range(3):  # Use 3 levels of the pyramid (adjust as needed)
        im1 = cv2.pyrDown(im1)
        im2 = cv2.pyrDown(im2)
        pyramid_im1.append(im1)
        pyramid_im2.append(im2)

    # Initialize variables to store the best alignment parameters and correlation score
    best_shift = (0, 0)
    best_score = 0.0

    # Iterate over the pyramid levels starting from the coarsest
    for level in range(len(pyramid_im1) - 1, -1, -1):
        scaled_im1 = pyramid_im1[level]
        scaled_im2 = pyramid_im2[level]

        # Determine the search range based on scaled image dimensions
        shift_range_x = int(scaled_im1.shape[1] * 0.1)  # 10% of scaled image width
        shift_range_y = int(scaled_im1.shape[0] * 0.1)  # 10% of scaled image height

        # Iterate over the range of possible shifts and calculate correlation scores
        for shift_x in range(-shift_range_x, shift_range_x + 1):
            for shift_y in range(-shift_range_y, shift_range_y + 1):
                print("in for loop...")
                # Apply the shift to scaled_im1
                shifted_im1 = np.roll(scaled_im1, shift_x, axis=1)
                shifted_im1 = np.roll(shifted_im1, shift_y, axis=0)

                # Calculate the correlation coefficient between shifted_im1 and scaled_im2
                correlation = np.corrcoef(shifted_im1.flatten(), scaled_im2.flatten())[0, 1]

                # Update the best alignment parameters if the correlation is higher
                if correlation > best_score:
                    best_score = correlation
                    best_shift = (shift_x * 2 ** level, shift_y * 2 ** level)  # Scale shift by pyramid level

    return best_shift


def findCorrelation(img1: np.ndarray, img2: np.ndarray):
    """
    This function looks for two points, one from @img1 and second from @img2.
    The two points are the ones with the highest correlation.
    :param img1: first image
    :param img2: second image
    :return: 2 points - x1, y1, x2, y2
    """
    # img1[img1 == 0] = np.float("inf")
    # img2[img2 == 0] = np.float("inf")
    img_shape = np.max(img1.shape) // 2
    im1FFT = np.fft.fft2(np.pad(img1, img_shape))
    im2FFT = np.fft.fft2(np.pad(img2, img_shape))
    prod = im1FFT * im2FFT.conj()
    res = np.fft.fftshift(np.fft.ifft2(prod))
    correlation = res.real[1 + img_shape:-img_shape + 1, 1 + img_shape:-img_shape + 1]
    p1y, p1x = np.unravel_index(np.argmax(correlation), correlation.shape)
    p2y, p2x = np.array(img2.shape) // 2
    return p1x, p1y, p2x, p2y


def findTranslationCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Translation.
    :return: Translation matrix by correlation.
    """
    X1, Y1, X2, Y2 = findCorrelation(im1, im2)
    return np.array([[1, 0, (X2 - X1 - 1)], [0, 1, (Y2 - Y1 - 1)], [0, 0, 1]], dtype=np.float)

def findRigidCorr(im1: np.ndarray, im2: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: image 1 after Rigid.
    :return: Rigid matrix by correlation.

"""

def estimate_rotation_translation(image1, image2):
    # Convert images to grayscale
    gray1 = image1
    gray2 = image2

    # Initialize SIFT detector
    sift = cv2.SIFT_create()

    # Detect keypoints and compute descriptors
    keypoints1, descriptors1 = sift.detectAndCompute(gray1, None)
    keypoints2, descriptors2 = sift.detectAndCompute(gray2, None)

    # Match keypoints
    matcher = cv2.BFMatcher()
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by distance
    matches = sorted(matches, key=lambda x: x.distance)

    # Select top matches
    num_matches = int(len(matches) * 0.15)
    matches = matches[:num_matches]

    # Get corresponding keypoints
    src_pts = np.float32([keypoints1[match.queryIdx].pt for match in matches]).reshape(-1, 1, 2)
    dst_pts = np.float32([keypoints2[match.trainIdx].pt for match in matches]).reshape(-1, 1, 2)

    # Estimate affine transformation matrix
    transformation_matrix, _ = cv2.estimateAffinePartial2D(src_pts, dst_pts)

    # Extract rotation angle and translation
    rotation_angle = np.arctan2(transformation_matrix[1, 0], transformation_matrix[0, 0]) * 180.0 / np.pi
    dx = transformation_matrix[0, 2]
    dy = transformation_matrix[1, 2]

    return rotation_angle, dx, dy



def warpImages(im1: np.ndarray, im2: np.ndarray, T: np.ndarray) -> np.ndarray:
    """
    :param im1: input image 1 in grayscale format.
    :param im2: input image 2 in grayscale format.
    :param T: is a 3x3 matrix such that each pixel in image 2
    is mapped under homogenous coordinates to image 1 (p2=Tp1).
    :return: warp image 2 according to T and display both image1
    and the wrapped version of the image2 in the same figure.
    """
    h, w = im2.shape
    X, Y = np.meshgrid(range(w), range(h))
    I_new = np.array([X.flatten(), Y.flatten(), np.ones_like(X.flatten())])

    I_old = (np.linalg.inv(T)) @ I_new
    mask1 = (I_old[0, :] > w) | (I_old[0, :] < 0)
    mask2 = (I_old[1, :] > h) | (I_old[1, :] < 0)
    I_old[0, :][mask1] =0
    I_old[1, :][mask2] =0
    new_img = im2[I_old[1, :].astype(int), I_old[0, :].astype(int)]
    
    return (new_img.reshape(im2.shape))

# ---------------------------------------------------------------------------
# --------------------- Gaussian and Laplacian Pyramids ---------------------
# ---------------------------------------------------------------------------


def gaussianPyr(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Gaussian Pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Gaussian pyramid (list of images)
    """
    pyramid = [img]  # Initialize the pyramid with the original image

    for level in range(1, levels):
        # Apply Gaussian blur to the previous level image
        prev_level = pyramid[level - 1]
        kernel_size = int(0.3 * ((5 - 1) * 0.5 - 1) + 0.8)
        kernel = cv2.getGaussianKernel(kernel_size, 0)
        blurred = cv2.filter2D(prev_level, -1, kernel)

        # Downsample the blurred image
        downsampled = cv2.pyrDown(blurred)

        # Add the downsampled image to the pyramid
        pyramid.append(downsampled)

    return pyramid


def laplaceianReduce(img: np.ndarray, levels: int = 4) -> List[np.ndarray]:
    """
    Creates a Laplacian pyramid
    :param img: Original image
    :param levels: Pyramid depth
    :return: Laplacian Pyramid (list of images)
    """
    # pyramid = []
    # for level in range(levels):
    #     if level == 0:
    #         blurred = cv2.GaussianBlur(img, (5, 5), 0)
    #         pyramid.append(img - blurred)
    #     else:
    #         downsampled = cv2.pyrDown(pyramid[level - 1])
    #         blurred = cv2.GaussianBlur(downsampled, (5, 5), 0)
    #         pyramid.append(downsampled - blurred)
    # return pyramid

    layer = img.copy()
    gaussian_pyramid = [layer]    #Gaussian Pyramid
    laplacian_pyramid = []         # Laplacian Pyramid
    for i in range(levels):
        blur = cv2.GaussianBlur(gaussian_pyramid[i], (5,5),5)
        laplacian = gaussian_pyramid[i]-blur
        width = int(blur.shape[1] / 2)
        height = int(blur.shape[0] / 2)
        layer = cv2.resize(blur,(width,height))
        gaussian_pyramid.append(layer)
        laplacian_pyramid.append(laplacian)
    gaussian_pyramid.pop(-1)
    return  laplacian_pyramid

def laplaceianExpand(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a Laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    levels = len(lap_pyr)
    img_reconstructed = lap_pyr[levels - 1]
    for level in range(levels - 2, -1, -1):
        expanded = cv2.pyrUp(img_reconstructed, dstsize=(lap_pyr[level].shape[1], lap_pyr[level].shape[0]))
        img_reconstructed = expanded + lap_pyr[level]
    img_reconstructed = np.clip(img_reconstructed, 0.0, 1.0)
    img_reconstructed = (img_reconstructed * 255).astype(np.uint8)

    return img_reconstructed
    
    
    
    for i in range((len(lap_pyr)-2), -1,-1):
        _h = lap_pyr[i+1].shape[1]
        _w = lap_pyr[i+1].shape[0]
        if i == len(lap_pyr)-2:
            G_rec = lap_pyr[i] + cv2.GaussianBlur(cv2.resize(lap_pyr[i+1], (int(_h*2), int(_w*2))),(5,5),1)
        else:
            G_rec = lap_pyr[i] + cv2.GaussianBlur(cv2.resize(G_rec, (int(G_rec.shape[1]*2),int(G_rec.shape[0]*2))),(5,5),1)
    G_rec = G_rec.astype(np.float32)
    G_rec = np.clip(G_rec, 0.0, 1.0)
    G_rec = (G_rec * 255).astype(np.uint8)
    return G_rec
    
    # levels = len(lap_pyr)
    # img_reconstructed = lap_pyr[levels - 1].astype(np.float32)
    # for level in range(levels - 2, -1, -1):
    #     expanded = cv2.pyrUp(img_reconstructed)
    #     img_reconstructed = expanded[:lap_pyr[level].shape[0], :lap_pyr[level].shape[1]] + lap_pyr[level]
    min_val = np.min(G_rec)
    max_val = np.max(G_rec)
    G_rec_normalized = (G_rec - min_val) / (max_val - min_val)

    img_reconstructed = np.clip(G_rec_normalized, 0.0, 1.0)
    img_reconstructed = (img_reconstructed * 255).astype(np.uint8)

    return img_reconstructed

def laplaceianExpand0(lap_pyr: List[np.ndarray]) -> np.ndarray:
    """
    Restores the original image from a laplacian pyramid
    :param lap_pyr: Laplacian Pyramid
    :return: Original image
    """
    levels = len(lap_pyr)
    img_reconstructed = lap_pyr[levels - 1]
    for level in range(levels - 2, -1, -1):
        expanded = cv2.pyrUp(img_reconstructed)
        img_reconstructed = expanded + lap_pyr[level]
    return img_reconstructed


def pyrBlend(img_1: np.ndarray, img_2: np.ndarray,
             mask: np.ndarray, levels: int) -> tuple[np.ndarray, np.ndarray]:
    """
    Blends two images using PyramidBlend method
    :param img_1: Image 1
    :param img_2: Image 2
    :param mask: Blend mask
    :param levels: Pyramid depth
    :return: (Naive blend, Blended Image)
    """
    assert img_1.shape == img_2.shape, "Images must have the same shape."
    assert img_1.shape[:2] == mask.shape[:2], "Mask size must match image size."


    # Generate Gaussian pyramids for the images and the mask
    img_1_pyr = [img_1]
    img_2_pyr = [img_2]
    mask_pyr = [mask]

    for level in range(levels - 1):
        img_1_pyr.append(cv2.pyrDown(img_1_pyr[level]))
        img_2_pyr.append(cv2.pyrDown(img_2_pyr[level]))
        mask_pyr.append(cv2.pyrDown(mask_pyr[level]))

    # Blend the images in the pyramid
    blended_pyr = []
    for level in range(levels):
        blended = img_1_pyr[level] * mask_pyr[level] + img_2_pyr[level] * (1 - mask_pyr[level])
        blended_pyr.append(blended)

    # Reconstruct the blended image from the pyramid
    blended_img = blended_pyr[levels - 1]
    for level in range(levels - 2, -1, -1):
        blended_img_up = cv2.pyrUp(blended_img)
        blended_img_up = blended_img_up[:blended_pyr[level].shape[0], :blended_pyr[level].shape[1]]
        blended_img = blended_img_up + blended_pyr[level]

    # Naive blend without using the pyramid
    naive_blend = img_1 * mask + img_2 * (1 - mask)

    return naive_blend, blended_img

