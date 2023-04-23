import cv2
import numpy as np
import os
from PIL import Image

img_path = "_lidar/00077821.png"
# img_path2 = "examples/PixelFormer/models/result_pixelformer_kittieigen/raw/2011_09_26_drive_0002_sync_image_0000000005_image_02.png"
image = Image.open(img_path)
img = cv2.imread(img_path)

cv2.imshow("", img)
cv2.waitKey(0)
cv2.destroyAllWindows()

print(image.mode)
print(f"Image resolution: {img.shape}")
print(f"Data type: {img.dtype}")
print(f"Min value: {np.min(img)}")
print(f"Max value: {np.max(img)}")

print(img[img > 76])

flag = 0
if flag == 1:
    save_path = "out_fov80/rangedo80"
    img_path = "out_fov80/depth"

    img_names = sorted(os.listdir(img_path))

    for filename in img_names:

        img = cv2.imread(os.path.join(img_path, filename), cv2.IMREAD_GRAYSCALE).astype(float)

        img_norm = img / 255.0

        img_scaled = (img_norm * 80.0) + 175

        img_uint8 = np.uint8(img_scaled)
        
        cv2.imwrite(os.path.join(save_path, filename), img_uint8)

if flag == 2:
    save_path = "out_fov80/rangedo80"
    img_path = "out_fov80/depth"

    img_names = sorted(os.listdir(img_path))

    for filename in img_names:

        img = cv2.imread(os.path.join(img_path, filename), cv2.IMREAD_GRAYSCALE).astype(float)

        img[img > 80] = 0

        img_uint8 = np.uint8(img)

        cv2.imwrite(os.path.join(save_path, filename), img_uint8)