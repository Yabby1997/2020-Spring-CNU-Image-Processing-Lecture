import numpy as np
import cv2

def my_padding(src, pad_shape, pad_type = 'zero'):
    (h, w) = src.shape
    (p_h, p_w) = pad_shape
    pad_img = np.zeros((h + 2 * p_h, w + 2 * p_w))
    pad_img[p_h:h + p_h, p_w:w + p_w] = src

    if pad_type == 'repetition':
        pad_img[ :p_h, p_w:p_w + w] = src[0, :]
        pad_img[p_h + h: , p_w:p_w + w] = src[h-1,:]
        pad_img[:,:p_w] = pad_img[:,p_w:p_w + 1]
        pad_img[:,p_w + w:] = pad_img[:,p_w + w - 1:p_w + w]

    else:
        pass

    return pad_img


if __name__=='__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    my_pad_img = my_padding(src, (20, 20), 'repetition')

    my_pad_img = (my_pad_img + 0.5).astype(np.uint8)
    cv2.imshow('original', src)
    cv2.imshow('my padding img', my_pad_img)

    cv2.waitKey()
    cv2.destroyAllWindows()


