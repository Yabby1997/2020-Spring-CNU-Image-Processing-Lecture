import cv2
import numpy as np


def dilation(B, S):
    S_h, S_w = S.shape
    img_dilation = np.copy(B)
    target_row, target_col = np.where(B==255)
    for row, col in zip(target_row, target_col):
        for i in range(S_h):
            for j in range(S_w):
                checking_row = row + i - S_h//2
                checking_col = col + j - S_w//2
                if checking_row >= 0 and checking_col >= 0:
                    try:
                        img_dilation[checking_row][checking_col] = S[i][j]
                    except IndexError:
                        pass

    return img_dilation


def erosion(B, S):
    S_h, S_w = S.shape
    img_erosion = np.copy(B)
    target_row, target_col = np.where(B==255)
    for row, col in zip(target_row, target_col):
        for i in range(S_h):
            for j in range(S_w):
                checking_row = row + i - S_h//2
                checking_col = col + j - S_w//2
                if checking_row >= 0 and checking_col >= 0:
                    try:
                        if B[checking_row][checking_col] != S[i][j]:
                            img_erosion[row][col] = 0
                    except IndexError:
                        img_erosion[row][col] = 0
                else:
                    img_erosion[row][col] = 0

    return img_erosion


def opening(B, S):
    temp_erosion = erosion(B, S)
    img_opening = dilation(temp_erosion, S)

    return img_opening


def closing(B, S):
    temp_dilation = dilation(B, S)
    img_closing = erosion(temp_dilation, S)

    return img_closing


if __name__ == '__main__':
    B = np.array(
        [[0, 0, 0, 0, 0, 0, 0, 0],
         [255, 255, 255, 255, 255, 255, 255, 0],
         [0, 0, 0, 255, 255, 255, 255, 0],
         [0, 0, 0, 255, 255, 255, 255, 0],
         [0, 0, 255, 255, 255, 255, 255, 0],
         [0, 0, 0, 255, 255, 255, 255, 0],
         [0, 0, 0, 0, 0, 0, 0, 0]]
    , dtype = np.uint8)

    S = np.array(
        [[255, 255, 255],
         [255, 255, 255],
         [255, 255, 255]]
    , dtype = np.uint8)

    cv2.imwrite('morphology_B.png', B)

    img_dilation = dilation(B, S)
    print(img_dilation)
    cv2.imwrite('morphology_dilation.png', img_dilation)

    img_erosion = erosion(B, S)
    print(img_erosion)
    cv2.imwrite('morphology_erosion.png', img_erosion)

    img_opening = opening(B, S)
    print(img_opening)
    cv2.imwrite('morphology_opening.png', img_opening)

    img_closing = closing(B, S)
    print(img_closing)
    cv2.imwrite('morphology_closing.png', img_closing)


