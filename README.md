# ImageProcessing-EX1

Python version :Python 3.10.4

platform: vscode

##the functions:


transformRGB2YIQ:

Converts an RGB image to YIQ color space, 

firstly,I normlize the image .

then multiply with the YIQ kernel matrix.


transformYIO2RGB:

Converts an YIQ image to RGB color space.

firstly, normlize.

then Compute the (multiplicative) inverse of a matrix. 

on that we do transpose and then multiply with the YIQ image 


Equalizing histograms:

Equalizes the histogram of an image.

steps:

firstly, if the image is not in gray scale , transform it to YIQ model and use only the Y channel

Calculate the image histogram (range = [0, 255])

Calculate the normalized Cumulative Sum (CumSum)

Create a LookUpTable(LUT), such that for each intensity i, LUT[i] = (CumSum[i]/allPixels) · 255

Replace each intesity i with LUT[i].



quantization :

Quantized an image in to **nQuant** colors:

reducing the number of distinct colors in an image to nQuant. 


gamma correction:

Gamma correction is a technique used in digital image processing to adjust the brightness and contrast of an image. It involves manipulating the gamma value, which is a parameter that describes the relationship between the pixel values in an image and their perceived brightness by the human eye.


Gamma correction is typically performed by raising the pixel values to a certain power (usually less than 1) to make the image appear brighter or darker. The gamma value used in this process depends on the display medium and the intended use of the image. For example, images intended for viewing on a computer monitor typically have a gamma value of around 2.2, while images intended for printing have a gamma value of around 1.8.




