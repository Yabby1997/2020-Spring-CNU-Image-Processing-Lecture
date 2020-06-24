import numpy as np
import cv2
import my_JPEG_DCT as md


def Quantization_Luminance():
    luminance = np.array(
        [[16, 11, 10, 16, 24, 40, 51, 61],
         [12, 12, 14, 19, 26, 58, 60, 55],
         [14, 13, 16, 24, 40, 57, 69, 56],
         [14, 17, 22, 29, 51, 87, 80, 62],
         [18, 22, 37, 56, 68, 109, 103, 77],
         [24, 35, 55, 64, 81, 104, 113, 92],
         [49, 64, 78, 87, 103, 121, 120, 101],
         [72, 92, 95, 98, 112, 100, 103, 99]])
    return luminance


def encoding(src, blocksize):
    h, w = src.shape
    encoded = np.array(list())
    for row in range(h // blocksize):
        for col in range(w // blocksize):
            quantized = np.round((src[row * blocksize:row * blocksize + blocksize, col * blocksize:col * blocksize + blocksize] / Quantization_Luminance()))
            encoded = np.concatenate((encoded, zigzag_encoding(quantized, blocksize)), axis=0)
    print('ENCODED DATA TOTAL LENGTH :', len(encoded))
    return encoded


def decoding(zigzag_encoded, blocksize):
    h, w = src.shape
    decoded = np.zeros((h, w))
    from_index = 0
    to_index = 0
    for row in range(h // blocksize):
        for col in range(w // blocksize):
            while not np.isnan(zigzag_encoded[to_index]) and to_index - from_index < blocksize**2 and to_index < len(zigzag_encoded):
                to_index += 1
            zigzag_decoded = zigzag_decoding(zigzag_encoded[from_index:to_index], blocksize)
            decoded[row * blocksize:row * blocksize + blocksize, col * blocksize:col * blocksize + blocksize] = zigzag_decoded * Quantization_Luminance()
            to_index += 1
            from_index = to_index
    return decoded


def zigzag_encoding(block, blocksize):
    zigzag_value = list()
    row, col, limit = 0, 0, 0
    zigzag_value.append(block[row][col])
    while row + 1 != blocksize or col + 1 != blocksize:
        if row == limit:
            if col + 1 == blocksize:
                row += 1
                limit += 1
                zigzag_value.append(block[row][col])
            else:
                col += 1
                zigzag_value.append(block[row][col])
            while col != limit:
                row += 1
                col -= 1
                zigzag_value.append(block[row][col])

        elif col == limit:
            if row + 1 == blocksize:
                col += 1
                limit += 1
                zigzag_value.append(block[row][col])
            else:
                row += 1
                zigzag_value.append(block[row][col])
            while row != limit:
                row -= 1
                col += 1
                zigzag_value.append(block[row][col])

    zigzag_value = np.array(zigzag_value)

    index = blocksize**2 - 1
    while zigzag_value[index] == 0:
        if index == 0:
            break;
        index -= 1

    result = zigzag_value[0:index + 1]
    return np.append(result, np.nan)


def zigzag_decoding(encoded, blocksize):
    block = np.zeros((blocksize, blocksize))
    if encoded.size != blocksize**2:
        decoded = np.append(encoded, np.zeros((1, blocksize**2 - encoded.size)))
    else:
        decoded = encoded

    index, row, col, limit = 0, 0, 0, 0
    block[row][col] = decoded[index]
    index += 1
    while row + 1 != blocksize or col + 1 != blocksize:
        if row == limit:
            if col + 1 == blocksize:
                row += 1
                limit += 1
                block[row][col] = decoded[index]
                index += 1
            else:
                col += 1
                block[row][col] = decoded[index]
                index += 1
            while col != limit:
                row += 1
                col -= 1
                block[row][col] = decoded[index]
                index += 1

        elif col == limit:
            if row + 1 == blocksize:
                col += 1
                limit += 1
                block[row][col] = decoded[index]
                index += 1
            else:
                row += 1
                block[row][col] = decoded[index]
                index += 1
            while row != limit:
                row -= 1
                col += 1
                block[row][col] = decoded[index]
                index += 1

    return block


def my_JPEG_encoding(src, block_size=8):
    subtracted_src = src - 128
    original_dct = md.my_DCT(subtracted_src, block_size, mode='normal')
    zigzag_value = encoding(original_dct, block_size)
    return zigzag_value


def my_JPEG_decoding(zigzag_value, block_size=8):
    damaged_dct = decoding(zigzag_value, block_size)
    restored_subtracted = md.my_DCT(damaged_dct, block_size, mode='inverse')
    restored = (restored_subtracted + 128).astype(np.uint8)
    return restored


if __name__ == '__main__':
    src = cv2.imread('Lena.png', cv2.IMREAD_GRAYSCALE)

    '''src = np.array(
        [[52, 55, 61, 66, 70, 61, 64, 73],
         [63, 59, 66, 90, 109, 85, 69, 72],
         [62, 59, 68, 113, 144, 104, 66, 73],
         [63, 58, 71, 122, 154, 106, 70, 69],
         [67, 61, 68, 104, 126, 88, 68, 70],
         [79, 65, 60, 70, 77, 68, 58, 75],
         [85, 71, 64, 59, 55, 61, 65, 83],
         [87, 79, 69, 68, 65, 76, 78, 94]])'''

    src = src.astype(np.float)
    zigzag_value = my_JPEG_encoding(src)
    print(zigzag_value[:10])

    dst = my_JPEG_decoding(zigzag_value)
    src = src.astype(np.uint8)
    cv2.imshow('original', src)
    cv2.imshow('result', dst)
    
    cv2.waitKey()
    cv2.destroyAllWindows()
