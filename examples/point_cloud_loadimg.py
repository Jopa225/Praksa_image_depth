import numpy as np
import open3d as o3d
import cv2
import matplotlib.pyplot as plt

# Load depth image into a numpy array
depth_image = plt.imread('examples/data_depth_selection/depth_selection/kitti/groundtruth_depth/2011_09_26_drive_0002_sync_groundtruth_depth_0000000005_image_02.png')
print(f"Image resolution: {depth_image.shape}")
print(f"Data type: {depth_image.dtype}")
print(f"Min value: {np.min(depth_image)}")
print(f"Max value: {np.max(depth_image)}")

depth_instensity = np.array(255 * depth_image, dtype=np.uint8)
print(np.min(depth_instensity))
plt.imshow(depth_instensity, cmap="gray")
plt.show()

image_h, image_w = depth_image.shape[0], depth_image.shape[1]
fov = 110.0
f_x = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
f_y = f_x * image_w / image_h
c_x = image_w / 2
c_y = image_h / 2

# Camera intrinsics matrix (can be obtained from CARLA)
intrinsics = np.array([[f_x, 0.0, c_x], [0.0, f_y, c_y], [0.0, 0.0, 1.0]])

# Depth-to-3D conversion formula
fx, fy, cx, cy = intrinsics[0, 0], intrinsics[1, 1], intrinsics[0, 2], intrinsics[1, 2]

c, r = np.meshgrid(np.arange(image_w), np.arange(image_h), sparse=True)

jj = np.tile(range(image_w), image_h)
ii = np.repeat(range(image_h), image_w)
xx = (jj - cx) / fx
yy = (ii - cy) / fy



# Create a point cloud from the 3D points
length = image_h * image_w
# depth_image = cv2.cvtColor(depth_image, cv2.COLOR_RGB2GRAY)
z = depth_image.reshape(image_h * image_w)
pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
print(pcd)
print(pcd.shape)
print(pcd.dtype)
cloud = o3d.geometry.PointCloud()
cloud.points = o3d.utility.Vector3dVector(pcd)

# Visualize the point cloud
o3d.visualization.draw_geometries([cloud])