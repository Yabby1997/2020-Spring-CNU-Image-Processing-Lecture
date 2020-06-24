import numpy as np
import cv2
import matplotlib.pyplot as plt


def my_calcHist(src):
    h, w = src.shape
    hist = np.zeros((256, ), dtype=np.int)
    for row in range(h):
        for col in range(w):
            hist[src[row, col]] += 1
    return hist


def my_normalize_hist(hist, pixel_num):
    normalized_hist = hist/pixel_num
    return normalized_hist


def my_PDF2CDF(pdf):
    cdf = np.zeros((256, ), dtype=np.float)
    cdf[0] = pdf[0]
    for i in range(1, len(pdf)):
        cdf[i] = cdf[i - 1] + pdf[i]
    return cdf


def my_denormalize(normalized, gray_level):
    denormalized = normalized * gray_level
    return denormalized


def my_calcHist_equalization(denormalized, hist):
    hist_equal = np.zeros((256, ), dtype=np.int)
    denormalized = (np.floor(denormalized)).astype(int)
    for i in range(len(denormalized)):
        hist_equal[denormalized[i]] += hist[i]
    return hist_equal


def my_equal_img(src, output_gray_level):
    h, w = src.shape
    dst = np.zeros((h, w), dtype=np.uint8)                          #int로 하면 오류발생. 이미지는 uint8 기억하기.
    for row in range(h):
        for col in range(w):
            dst[row, col] = output_gray_level[src[row, col]]
    return dst


#input_image의  equalization된 histogram & image 를 return
def my_hist_equal(src):
    (h, w) = src.shape
    max_gray_level = 255
    histogram = my_calcHist(src)
    normalized_histogram = my_normalize_hist(histogram, h * w)
    normalized_output = my_PDF2CDF(normalized_histogram)
    denormalized_output = my_denormalize(normalized_output, max_gray_level)
    output_gray_level = denormalized_output.astype(int)
    hist_equal = my_calcHist_equalization(output_gray_level, histogram)
    
    dst = my_equal_img(src, output_gray_level)

    return dst, hist_equal


if __name__ == '__main__':
    src = cv2.imread('fruits_div3.jpg', cv2.IMREAD_GRAYSCALE)
    hist = my_calcHist(src)
    dst, hist_equal = my_hist_equal(src)

    cv2.imshow('original', src)
    binX = np.arange(len(hist))
    plt.title('my histogram')
    plt.bar(binX, hist, width=0.5, color='g')
    plt.show()

    cv2.imshow('equalization after image', dst)
    binX = np.arange(len(hist_equal))
    plt.title('my histogram equalization')
    plt.bar(binX, hist_equal, width=0.5, color='g')
    plt.show()

    cv2.waitKey()
    cv2.destroyAllWindows()