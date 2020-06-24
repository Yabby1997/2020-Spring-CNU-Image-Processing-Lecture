import cv2
import numpy as np

def my_filtering(src, ftype, fsize):
    (h, w) = src.shape
    dst = np.zeros((h, w))

    fh = fsize[0]
    fw = fsize[1]
    divisor = fh * fw
    center_row = int(fh / 2)
    center_col = int(fw / 2)
    row_iteration = h - fh + 1
    col_iteration = w - fw + 1

    if ftype == 'average':
        print('average filtering')
        mask = np.ones(fsize, dtype=float) / divisor
        print(mask)

    elif ftype == 'sharpening':
        print('sharpening filtering')
        mask = -1 * (np.ones(fsize, dtype=float) / divisor)
        mask[center_row][center_col] += 2
        print(mask)

    for row in range(row_iteration):
        for col in range(col_iteration):
            target_area = src[row:row+fh, col:col+fw]
            filtered_value = np.sum(target_area * mask)
            if filtered_value < 0:
                filtered_value = 0
            elif filtered_value > 255:
                filtered_value = 255
            dst[row + center_row][col + center_col] = filtered_value

    dst = (dst + 0.5).astype(np.uint8)

    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    # 3x3 filter
    #dst_average = my_filtering(src, 'average', (3, 3))
    #dst_sharpening = my_filtering(src, 'sharpening', (3, 3))

    # do as you want
    dst_average = my_filtering(src, 'average', (1024, 1024))
    dst_sharpening = my_filtering(src, 'sharpening', (1024, 1024))

    # 11x13 filter
    #dst_average = my_filtering(src, 'average', (11,13))
    #dst_sharpening = my_filtering(src, 'sharpening', (11,13))

    cv2.imshow('original', src)
    cv2.imshow('average filter', dst_average)
    cv2.imshow('sharpening filter', dst_sharpening)
    cv2.waitKey()
    cv2.destroyAllWindows()
