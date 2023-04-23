import numpy as np
import open3d as o3d


pcd = o3d.geometry.PointCloud()

# the method Vector3dVector() will convert numpy array of shape (n, 3) to Open3D format.
# see http://www.open3d.org/docs/release/python_api/open3d.utility.Vector3dVector.html#open3d.utility.Vector3dVector
# pcd.points = o3d.utility.Vector3dVector(pts)

# read ply file
pcd = o3d.io.read_point_cloud('examples/KITTI_Dataset_CARLA_v0.9.14/Carla/Maps/Town01/georef_colorized/point_cloud_rgb_00.ply')

# visualize
o3d.visualization.draw_geometries([pcd])
