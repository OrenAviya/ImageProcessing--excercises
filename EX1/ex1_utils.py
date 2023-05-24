"""
        '########:'##::::'##::::'##:::
         ##.....::. ##::'##:::'####:::
         ##::::::::. ##'##::::.. ##:::
         ######:::::. ###::::::: ##:::
         ##...:::::: ## ##:::::: ##:::
         ##:::::::: ##:. ##::::: ##:::
         ########: ##:::. ##::'######:
        ........::..:::::..:::......::
"""
from typing import List
import numpy as np
import cv2 
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


LOAD_GRAY_SCALE = 1
LOAD_RGB = 2

YIQ_kernel = np.array([[0.299, 0.587, 0.114], [0.596, -0.275, -0.321], [0.212, -0.523, 0.311]])


def myID() -> np.int:
    """
    Return my ID (not the friend's ID I copied from)
    :return: int
    """
    return 322273301


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename)
    if representation == LOAD_GRAY_SCALE:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
       # img = img[:, :, 1]
    else:
        img = cv2.cvtColor(img , cv2.COLOR_BGR2RGB)
    
    img = cv2.normalize(img, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    return img
    
        
    cv2.imshow('image',img)
    return img


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
     

    img = imReadAndConvert(filename, representation)   
    
    if representation == LOAD_GRAY_SCALE:
        plt.imshow(img, cmap="gray")  # display gray image
    else:
        plt.imshow(img)

    plt.show()
    pass


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    #normlize the image to float :
    imgRGB = cv2.normalize(imgRGB, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    realShape = imgRGB.shape
    # multiply with the YIQ kernel matrix :
    YIQ = np.dot(imgRGB, YIQ_kernel.transpose())
    #back to normal shape:
    #YIQ.reshape(realShape)
    return YIQ
    pass


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    imgYIQ = cv2.normalize(imgYIQ, None, 0, 1, cv2.NORM_MINMAX, cv2.CV_64F)
    
    #linalg.inv - Compute the (multiplicative) inverse of a matrix. 
    # on that we do transpose and then multiply with
    # the YIQ image 
    imgInverse = (np.linalg.inv(YIQ_kernel)) 
    RGB = np.dot(imgYIQ, (imgInverse.transpose()))

    return RGB
    pass


def hsitogramEqualize(imgOrig: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return : (imgEq,histOrg,histEQ)

    """
    """Steps:
    • Calculate the image histogram (range = [0, 255])
    • Calculate the normalized Cumulative Sum (CumSum)
    • Create a LookUpTable(LUT), such that for each intensity i, LUT[i] = 
    (CumSum[i]/allPixels) · 255
    • Replace each intesity i with LUT[i]
    """
    image = imgOrig.copy()
    realShape = np.array(imgOrig).shape
    RGBSize =3 
    numToNorm = 255
    rgb = (len(realShape) == RGBSize)
    # if the image is not in gray scale , transform it to YIQ model and use only the Y channel
    if rgb:
        YIQImage = transformRGB2YIQ(image)
        image = YIQImage[:,:, 0]
    # convert to int before 
    image = (image * numToNorm).astype('uint32')  
    # calculate histogram
    histOrg, bins = np.histogram(image.flatten(), bins=256, range=[0, 255])
    CSOrg = histOrg.cumsum() 
    LUT = numToNorm * (CSOrg / CSOrg.max())
    # get the new image with the equalized histogram
    imEq = LUT[image.flatten()].reshape(realShape[0], realShape[1]).astype('uint32') 
    histEQ, new_bins = np.histogram(imEq, bins=256, range=[0, 255])
    # transform back to RGB and get all the channels YIQ
    if rgb :
        imEq = imEq / numToNorm
        YIQImage[:, :, 0] = imEq
        imEq = transformYIQ2RGB(YIQImage)
        # imEq = imEq.astype('float64')

    return imEq, histOrg, histEQ
    pass


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> tuple[List[np.ndarray], List[float]]:
    """
        Quantized an image in to **nQuant** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    numToNorm = 255
    upperBound = 256
    RGBSize = 3
    realShape = imOrig.shape
    rgb = len(realShape) == RGBSize
    img = imOrig.copy()
    
    if (rgb):
        YIQ = transformRGB2YIQ(imOrig)
        img = YIQ[:, :, 0]
    
     # make the fist z boundaries 
    pixels_ForEach_Range = num_of_pixels(img)/ nQuant
    img = (img * numToNorm).astype('uint32')
    hisOrg, bins = np.histogram(img, bins=256, range=[0, 255])
    curr_num_of_pixels = 0
    z_indexes = [0]  # indexes that corresponds to all z_i [z_0=0,z_1,...,z_k=255]
    for i in range(numToNorm + 1):
        if curr_num_of_pixels + hisOrg[i] < pixels_ForEach_Range:
            curr_num_of_pixels += hisOrg[i]
            if i == numToNorm:
                z_indexes.append(i)
        else:
            z_indexes.append(i)
            curr_num_of_pixels = hisOrg[i] 
    
    q = get_q_arr_from_z(z_indexes, hisOrg)
    z_ =np.asarray( [0] + [(q[i] + q[i + 1]) / 2 for i in range(len(q) - 1)] + [255]).astype('uint32')
    prev_z_arr = np.array(z_)
    
    error = []
    quaImg = []
    count = 0
# the two steps above nIter times: 1.Finding z   2.Finding q for *Optimal* image quantization 
    while (count < nIter):
        q = get_q_arr_from_z(z_, hisOrg)
        err = 0
        for k in range(len(z_) - 1):
            # over on z and calculate the error
            for j in range(z_[k]+1, z_[k + 1]+1):
                err += ((q[k] - j) ** 2) * (hisOrg[j]/num_of_pixels(imOrig))  
        # save all the errors
        error.append(np.min(err) / np.size(img))
        z_ = np.asarray( [0] + [(q[i] + q[i + 1]) / 2 for i in range(len(q) - 1)] + [255]).astype('uint32')
        if (z_ == prev_z_arr).all():
            break
        prev_z_arr = z_
        count += 1
        
    #LookUpTable to each 
    LUT = np.zeros(numToNorm + 1)
    LUT[0] = np.floor(q[0])
    for m in range(len(q)):
        lower_bound = np.floor(z_[m] + 1).astype(np.int64)
        upper_bound = np.floor(z_[m + 1]).astype(np.int64)
        LUT[lower_bound:upper_bound + 1] = (np.floor(q[m]))
    
    print(np.array(LUT))
    
    Ytemp = np.array(LUT)[img].reshape(realShape[0], realShape[1])/ numToNorm
    if rgb:  # transform back to RGB and get all the channels YIQ
        YIQtemp = YIQ.copy()
        YIQtemp[:, :, 0] =Ytemp
        YIQtemp = transformYIQ2RGB(YIQtemp)
        Ytemp = YIQtemp
        #quaImg = YIQtemp
    
    # else: 
    #     quaImg = LUT[imOrig.astype(np.int64)] / numToNorm
    #     quaImg.reshape(realShape[0], realShape[1])
    #quaImg = Ytemp
    quaImg.append(Ytemp)

    return quaImg, error
    pass


"""
    calculate the color of each boundaries
    :Param z: vector of boundaries
    :Param hisOrg: vector of original image histogram 
    :Return vector of color
    
"""
def meanIn(z: np.ndarray, hisOrg: np.ndarray) -> np.array:
    q = []
    for i in range(len(z) - 1):
        place = hisOrg[z[i]:z[i + 1]]
        q.append((place * range(z[i], z[i + 1])).sum() // place.sum())

    return q

def get_q_arr_from_z(z_arr: np.ndarray, hist: np.ndarray):
    """
    calculates q_i given z_i using weighted arithmetic mean
    :param z_arr: all indexes
    :param hist: image histogram
    :return: q_i
    """
    q_arr = []
    for i in range(len(z_arr) - 1):
        lower_bound = int(np.floor(z_arr[i]) + 1)
        upper_bound = int(np.floor(z_arr[i + 1]))
        if lower_bound != upper_bound:
            weights = np.array([hist[g] for g in range(lower_bound, upper_bound + 1)])
            q_i = np.average(np.arange(lower_bound, upper_bound + 1), weights=weights)  # mean
            q_arr.append(q_i)
    return q_arr

def num_of_pixels(image):
    """
    :param image: some image
    :return:the dimension of the image
    """
    return image.shape[0] * image.shape[1]
