import cv2
import numpy as np
import math
import my_padding as my_p


def my_filtering(src, mask, pad_type='zero'):
    h, w = src.shape
    m_h, m_w = mask.shape

    pad_img = my_p.my_padding(src, (m_h // 2, m_w // 2), pad_type)

    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)

    return dst


def apply_lowNhigh_pass_filter(src, fsize, sigma=1, pad_type='zero'):
    y, x = np.mgrid[-(fsize // 2):(fsize // 2) + 1, -(fsize // 2):(fsize // 2) + 1]
    DoG_x = (-x / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))
    DoG_y = (-y / sigma ** 2) * np.exp(-(x ** 2 + y ** 2) / (2 * sigma ** 2))

    Ix = my_filtering(src, DoG_x, 'repetition')
    Iy = my_filtering(src, DoG_y, 'repetition')

    return Ix, Iy


def calcMagnitude(Ix, Iy):
    magnitude = np.sqrt(Ix ** 2 + Iy ** 2)

    return magnitude


def calcAngle(Ix, Iy):
    angle = np.arctan(Iy / Ix)

    return angle


def non_maximum_suppression(magnitude, angle):
    (h, w) = magnitude.shape
    larger_magnitude = np.zeros((h, w))

    for row in range(h):
        for col in range(w):
            inclination = np.tan(angle[row][col])
            best = magnitude[row][col]

            row_intercept = row - inclination * col

            col1 = max(col - 1, 0)
            col2 = min(col + 1, w - 1)
            row1 = max(row - 1, 0)
            row2 = min(row + 1, h - 1)

            if -1 <= inclination <= 1:
                row1 = inclination * col1 + row_intercept
                row2 = inclination * col2 + row_intercept

                try:
                    lower = math.floor(row1) + 1
                    upper = lower - 1
                    row_distance = min(row1 - upper, 1)
                    candidate1 = magnitude[lower][col1] * row_distance + magnitude[upper][col1] * (1 - row_distance)
                except IndexError:
                    candidate1 = 0

                try:
                    lower = min(math.floor(row2) + 1, h - 1)
                    upper = lower - 1
                    row_distance = min(row2 - upper, 1)
                    candidate2 = magnitude[lower][col2] * row_distance + magnitude[upper][col2] * (1 - row_distance)
                except IndexError:
                    candidate1 = 0

            else:
                col1 = (row1 - row_intercept) / inclination
                col2 = (row2 - row_intercept) / inclination

                try:
                    right = min(math.floor(col1) + 1, w - 1)
                    left = right - 1
                    col_distance = min(col1 - left, 1)
                    candidate1 = magnitude[row1][right] * col_distance + magnitude[row1][left] * (1 - col_distance)
                except IndexError:
                    candidate1 = 0

                try:
                    right = min(math.floor(col2) + 1, w - 1)
                    left = right - 1
                    col_distance = min(col2 - left, 1)
                    candidate2 = magnitude[row2][right] * col_distance + magnitude[row2][left] * (1 - col_distance)
                except IndexError:
                    candidate2 = 0

            if candidate1 > best or candidate2 > best:
                best = 0

            larger_magnitude[row][col] = best

            print('current : %d, %d\t value : %d' % (row, col, magnitude[row][col]))
            print('inclination(a) : %.5f\t intercept(b) : %.5f' % (inclination, row_intercept))
            print('candidate1 : %.5f, %.5f\t value : %.5f' % (row1, col1, candidate1))
            print('candidate2 : %.5f, %.5f\t value : %.5f' % (row2, col2, candidate2))
            print('selection :', best)
            print('#' * 60)

    magnitude = (magnitude / np.max(magnitude) * 255).astype(np.uint8)
    larger_magnitude = (larger_magnitude / np.max(larger_magnitude) * 255).astype(np.uint8)
    return magnitude, larger_magnitude


def double_thresholding(src):
    (h, w) = src.shape
    dst = np.zeros((h, w))

    high, _ = cv2.threshold(src, 0, 255, cv2.THRESH_OTSU)
    low = 0.4 * high

    for row in range(h):
        for col in range(w):
            if src[row][col] >= high:
                dst[row][col] = 255
            elif src[row][col] < low:
                dst[row][col] = 0
            else:
                dst[row][col] = -1

    group = 300
    weak_row, weak_col = np.where(dst == -1)
    for row, col in zip(weak_row, weak_col):
        if dst[row][col] == -1:
            dst[row][col] = group
            dst = check8neighbor_recursively(dst, row, col, group)
            group += 1
        print('[LABELING] group with %d, %d => %d' % (row, col, dst[row][col]))

    labeled_dst = dst

    for label in range(300, group):
        label_row, label_col = np.where(dst == label)
        for row, col in zip(label_row, label_col):
            for i in range(-1, 2):
                for j in range(-1, 2):
                    if i!= 0 or j != 0:
                        checking_row = row + i
                        checking_col = col + j
                        try:
                            if dst[checking_row][checking_col] >= 255:
                                dst = np.where(dst == label, dst[checking_row][checking_col], dst)
                                print('[WEAK MERGE] label %d meets %d, %d which is %d' % (label, checking_row, checking_col, dst[checking_row][checking_col]))
                        except IndexError:
                            pass

    merged_dst = dst

    dst = np.where(dst >= 300, 0, dst)

    return dst


def check8neighbor_recursively(dst, row, col, label):
    for i in range(-1, 2):
        for j in range(-1, 2):
            if i != 0 or j != 0:
                checking_row = row + i
                checking_col = col + j
                try:
                    if dst[checking_row][checking_col] == -1:
                        dst[checking_row][checking_col] = label
                        return check8neighbor_recursively(dst, checking_row, checking_col, label)
                except IndexError:
                    pass
    return dst


def my_canny_edge_detection(src, fsize=5, sigma=1, pad_type='zero'):
    Ix, Iy = apply_lowNhigh_pass_filter(src, fsize, sigma, pad_type)

    magnitude = calcMagnitude(Ix, Iy)
    angle = calcAngle(Ix, Iy)

    magnitude, larger_magnitude = non_maximum_suppression(magnitude, angle)

    dst = double_thresholding(larger_magnitude)

    return magnitude, larger_magnitude, dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    magnitude, larger_magnitude, dst = my_canny_edge_detection(src)

    cv2.imshow('original', src)
    cv2.imshow('magnitude', magnitude)
    cv2.imshow('larger magnitude', larger_magnitude)
    cv2.imshow('my canny edge detection', dst)
    cv2.waitKey()
    cv2.destroyAllWindows()
