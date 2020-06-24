import cv2
import math
import numpy as np


def my_bilinear(src, size):
    h, w = src.shape
    h_dst = size[0]
    w_dst = size[1]
    scale = h_dst / h

    dst = np.zeros((h_dst, w_dst), np.uint8)

    for row in range(h_dst):
        for col in range(w_dst):
            scaled_row = row / scale
            scaled_col = col / scale
            lower = min(math.floor(scaled_row) + 1, h - 1)
            upper = lower - 1
            right = min(math.floor(scaled_col) + 1, w - 1)
            left = right - 1
            row_distance = min(scaled_row - upper, 1)
            col_distance = min(scaled_col - left, 1)

            print('==' * 10)
            print('dst :', row, col)
            print('scaled row :', scaled_row)
            print('scaled col :', scaled_col)
            print('src upper :', upper)
            print('src lower :', lower)
            print('src left :', left)
            print('src right :', right)
            print('row distance :', row_distance)
            print('col distance :', col_distance)

            dst[row, col] = row_distance * col_distance * src[lower, right]\
                            + row_distance * (1 - col_distance) * src[lower, left]\
                            + (1 - row_distance) * col_distance * src[upper, right]\
                            + (1 - row_distance) * (1 - col_distance) * src[upper, left]

    print(dst)
    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    my_dst_mini = my_bilinear(src, (128, 128))
    my_dst = my_bilinear(my_dst_mini, (512, 512))

    cv2.imshow('original', src)
    cv2.imshow('my bilinear mini', my_dst_mini)
    cv2.imshow('my bilinear', my_dst)

    cv2.waitKey()
    cv2.destroyAllWindows()

