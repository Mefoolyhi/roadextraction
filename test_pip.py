import numpy as np
import cv2
import operator
from tqdm import tqdm

tile_size = 1024
step = tile_size

img = cv2.imread('kolomn.png')
img = np.asarray(img)
img = np.moveaxis(img, -1, 0)
_, w, h = img.shape

print(f'img shape is {img.shape}')
res_shape = tuple(map(operator.add, img.shape, (0, tile_size - w % tile_size, tile_size - h % tile_size)))
res = np.zeros(res_shape, dtype=np.float32)
print(res_shape)
res[:, 0:w, 0:h] = img
img = res

w_new = res_shape[1]
h_new = res_shape[2]
res = np.zeros((1, w_new, h_new), dtype=np.float32)
i = j = 0
with tqdm(total=(w // step + 1) * (h // step + 1)) as pbar:
    while i + tile_size <= w_new:
        j = 0
        while j + tile_size <= h_new:
            print(f'i is {i} j is {j}')
            frag = img[:, i:i+tile_size, j:j+tile_size]

            out = frag
            cv2.imwrite(f'frag{i},{j}.png', np.uint8(frag))
            res[:, i:i+tile_size, j:j+tile_size] = out[0]
            j += step
            pbar.update(1)
        i += step
max = np.max(res)
min = np.min(res)

img = res.reshape((w_new, h_new))
img = img[0:w, 0:h]
img = 255.0*(img - min)/(max-min)
print(f'result shape is {img.shape}')
cv2.imwrite(f'result.png', np.uint8(img))