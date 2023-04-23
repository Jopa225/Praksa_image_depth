#!/usr/bin/env python

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

import random
import time
import numpy as np
import cv2

IM_WIDTH = 1216
IM_HEIGHT = 352

def process_img(image):
    i = np.array(image.raw_data)
    i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    i3 = i2[:, :, :3]
    cv2.imshow("", i3)
    cv2.waitKey(1)
    return i3
    # return i3/255.0


def main():
    actor_list = []

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)

        world = client.get_world()

        blueprint_library = world.get_blueprint_library()

        bp = random.choice(blueprint_library.filter('vehicle'))
        if bp.has_attribute('color'):
            color = random.choice(bp.get_attribute('color').recommended_values)
            bp.set_attribute('color', color)
        
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)
        vehicle.set_autopilot(True)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        # change the dimensions of the image
        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '80')
        # camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        
        

        camera_bp1 = blueprint_library.find('sensor.camera.depth')
        camera_bp1.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp1.set_attribute('image_size_y', f'{IM_HEIGHT}')
        # camera_bp1.set_attribute('fov', '110')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera_depth = world.spawn_actor(camera_bp1, camera_transform, attach_to=vehicle)
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        actor_list.append(camera_depth)
        print('created %s' % camera.type_id)
        print('created %s' % camera_depth.type_id)

        cc = carla.ColorConverter.Depth
        # camera.listen(lambda data: process_img(data))

        camera.listen(lambda image: image.save_to_disk('out_fov80/rgb/%06d.png' % image.frame))
        camera_depth.listen(lambda image1: image1.save_to_disk('out_fov80/depth/%06d.png' % image1.frame, cc))
        
        time.sleep(30)


    finally:

        print('destroying actors')
        for actor in actor_list:
            actor.destroy()
        print('done.')


if __name__ == '__main__':

    main()