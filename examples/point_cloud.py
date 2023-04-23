#!/usr/bin/env python

# Copyright (c) 2019 Computer Vision Center (CVC) at the Universitat Autonoma de
# Barcelona (UAB).
#
# This work is licensed under the terms of the MIT license.
# For a copy, see <https://opensource.org/licenses/MIT>.

import glob
import os
import sys

try:
    sys.path.append(glob.glob('../carla/dist/carla-*%d.%d-%s.egg' % (
        sys.version_info.major,
        sys.version_info.minor,
        'win-amd64' if os.name == 'nt' else 'linux-x86_64'))[0])
except IndexError:
    pass

import carla
import time
import random
import numpy as np
import open3d as o3d

IM_WIDTH = 1216
IM_HEIGHT = 352

def process_depth_data(data):
    image = np.frombuffer(data.raw_data, dtype=np.dtype('f4'))
    image = np.reshape(image, (data.height, data.width))

    # convert depth image to point cloud
    image_h, image_w = image.shape
    fov = 80.0
    f_x = image_w / (2.0 * np.tan(fov * np.pi / 360.0))
    f_y = f_x * image_w / image_h
    c_x = image_w / 2.0
    c_y = image_h / 2.0
    # x, y = np.meshgrid(np.arange(image_w), np.arange(image_h), sparse=True)
    jj = np.tile(range(image_w), image_h)
    ii = np.repeat(range(image_h), image_w)
    xx = (jj - c_x) / f_x
    yy = (ii - c_y) / f_y

    length = image_h * image_w
    z = image.reshape(image_h * image_w)
    pcd = np.dstack((xx * z, yy * z, z)).reshape((length, 3))
    point_cloud = o3d.geometry.PointCloud()
    point_cloud.points = o3d.utility.Vector3dVector(pcd)

    o3d.visualization.draw_geometries([point_cloud])
    

def main():
    actor_list = []
    try:
        # connect to the CARLA simulator
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        # get the world and blueprint library
        world = client.get_world()
        blueprint_library = world.get_blueprint_library()

        bp = random.choice(blueprint_library.filter('vehicle'))

        #  initial transform to the vehicle
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)

        # get the camera sensor actor
        camera_bp = blueprint_library.find('sensor.camera.depth')
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '80')
        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)


        # Build the K projection matrix:
        # K = [[Fx,  0, image_w/2],
        #      [ 0, Fy, image_h/2],
        #      [ 0,  0,         1]]
        

        # In this case Fx and Fy are the same since the pixel aspect
        # ratio is 1
        #K = np.identity(3)
        #K[0, 0] = K[1, 1] = focal
        #K[0, 2] = image_w / 2.0
        #K[1, 2] = image_h / 2.0
        cc = carla.ColorConverter.LogarithmicDepth
        camera.listen(lambda image: image.save_to_disk('out_fov80/%06d.png' % image.frame, cc))
        #camera_settings = camera.get_blueprint().get_attribute('sensor_tick').recommended_values[0]
        #camera_settings['sensor_tick'] = '0.05'
        #camera.set_attribute('sensor_tick', str(camera_settings['sensor_tick']))

        # point cloud
        



        time.sleep(0)
    
    finally:
        print('destroying actors')
        camera.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()

