import numpy as np
import cv2
import my_padding as my_p


def my_get_Gaussian2D_mask(msize, sigma=1):
    y, x = np.mgrid[-(msize // 2):(msize // 2) + 1, -(msize // 2):(msize // 2) + 1]
    #2차 gaussian mask 생성
    gaus2D =   1 / (2 * np.pi * sigma**2) * np.exp(-(( x**2 + y**2 )/(2 * sigma**2)))
    #mask의 총 합 = 1
    gaus2D /= np.sum(gaus2D)
    print(gaus2D)
    return gaus2D


def my_filtering(src, mask, pad_type='zero'):
    (h, w) = src.shape
    (m_h, m_w) = mask.shape
    pad_img = my_p.my_padding(src, (m_h // 2, m_w // 2), pad_type)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            dst[row, col] = np.sum(pad_img[row:row + m_h, col:col + m_w] * mask)
    return dst


def my_normalize(src):
    dst = src.copy()
    dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)


def my_bilateral(src, msize, sigma, sigma_r, pad_type='zero'):
    (h, w) = src.shape
    pad_img = my_p.my_padding(src, (msize // 2, msize // 2), pad_type)
    dst = np.zeros((h, w))
    for row in range(h):
        for col in range(w):
            mask = np.zeros((msize, msize))

            for m_row in range(msize):
                for m_col in range(msize):
                    row_distance = msize//2 - m_row
                    col_distance = msize//2 - m_col
                    src_value = src[row][col]
                    target_value = pad_img[row + m_row][col + m_col]
                    mask[m_row][m_col] = np.exp(-((row_distance**2 + col_distance**2) / (2 * (sigma ** 2)))) * np.exp(-(((src_value - target_value) ** 2) / (2 * (sigma_r ** 2))))

            if row == 51 and col == 121:
                print('DONE!')
                mask_img = cv2.resize(mask, (200, 200), interpolation=cv2.INTER_NEAREST)
                mask_img = my_normalize(mask_img)
                img = pad_img[row:row + msize, col:col + msize]
                img = cv2.resize(img, (2000, 20000), interpolation=cv2.INTER_NEAREST)
                img = my_normalize(img)
                cv2.imshow('mask', mask_img)
                cv2.imshow('src', img)
                cv2.waitKey()
                cv2.destroyAllWindows()

            dst[row][col] = np.sum(pad_img[row:row + msize, col:col + msize] * mask) / np.sum(mask)
    dst = my_normalize(dst)
    return dst


if __name__ == '__main__':
    src = cv2.imread('Penguins_noise.png', cv2.IMREAD_GRAYSCALE)
    dst = my_bilateral(src, 5, 5, 30, 'repetition')
    dst = my_normalize(dst)

    gaus2D = my_get_Gaussian2D_mask(5 , sigma=1)
    print(gaus2D)
    dst_gaus2D= my_filtering(src, gaus2D)
    dst_gaus2D = my_normalize(dst_gaus2D)

    cv2.imshow('original', src)
    cv2.imshow('my gaussian', dst_gaus2D)
    cv2.imshow('my bilateral', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()

