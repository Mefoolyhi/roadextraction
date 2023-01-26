import cv2
import numpy as np


def one_hot_encode(label, label_values):
    semantic_map = []
    for colour in label_values:
        equality = np.equal(label, colour)
        class_map = np.all(equality, axis=-1)
        semantic_map.append(class_map)
    semantic_map = np.stack(semantic_map, axis=-1)
    return semantic_map


def reverse_one_hot(image):
    x = np.argmax(image, axis=-1)
    return x


def colour_code_segmentation(image, label_values):
    colour_codes = np.array(label_values)
    x = colour_codes[image.astype(int)]
    print(x)
    return x


class_rgb_values = [[0, 0, 0], [110, 110, 110]]
mask = cv2.cvtColor(cv2.imread('mask.tif'), cv2.COLOR_BGR2RGB)[0:1024, 0:1024, :]
print(mask.shape)
mask = one_hot_encode(mask, class_rgb_values).astype('float')
print(mask.shape)
roh = reverse_one_hot(mask)
print(roh.shape)
cv2.imwrite('reverse_one_hot.png', roh)
result = colour_code_segmentation(roh, class_rgb_values)
print(result.shape)
cv2.imwrite('colour_code_segmentation.png', result)
