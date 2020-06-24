import cv2
import numpy as np


def getC(i, n):
    if i == 0:
        return np.sqrt(1/n)
    else:
        return np.sqrt(2/n)


def my_normalize(src):
    dst = src.copy()
    if np.min(dst) != np.max(dst):
        dst = dst - np.min(dst)
    dst = dst / np.max(dst) * 255
    return dst.astype(np.uint8)


def my_DCT(src, n=8):
    h, w = src.shape
    dst = np.zeros((h, w), dtype=np.float)
    cosmask = np.zeros((n, n, n, n))

    for u in range(n):
        for v in range(n):
            for x in range(n):
                for y in range(n):
                    cosmask[u][v][x][y] = np.cos(((2 * x + 1) * u * np.pi) / (2 * n)) * np.cos(((2 * y + 1) * v * np.pi) / (2 * n))

    for row in range(h // n):
        for col in range(w // n):
            block = src[row * n:row * n + n, col * n:col * n + n]
            for u in range(n):
                for v in range(n):
                    dst[row * n + u][col * n + v] = getC(u, n) * getC(v, n) * np.sum(block * cosmask[u][v])

    print(dst)
    return dst


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)
    dst = my_DCT(src, 8)

    dst = my_normalize(dst)
    cv2.imshow('my DCT', dst)

    cv2.waitKey()
    cv2.destroyAllWindows()


