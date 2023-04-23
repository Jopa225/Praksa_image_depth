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



import torch
import carla
import random
import time
import numpy as np
import cv2
from queue import Queue
from queue import Empty
# from networks.PixelFormer import PixelFormer

# IM_WIDTH = 1216
# IM_HEIGHT = 352
IM_WIDTH = 640
IM_HEIGHT = 480

def process_img(data, queue):
    #i = np.array(data.raw_data, dtype = np.uint8)
    ##i2 = i.reshape((IM_HEIGHT, IM_WIDTH, 4))
    #i3 = np.delete(i2, 3, axis=2)

    queue.put(data)
    #if "window" not in globals():
    #    cv2.namedWindow("window")
    
    #cv2.imshow("window", i3)
    #cv2.waitKey(1)
    #return i3
    # return i3/255.0


def main():
    actor_list = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    

    try:
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        original_settings = world.get_settings()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 3.0
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # blueprints of type 'vehicle' 
        bp = random.choice(blueprint_library.filter('vehicle'))

        #  initial transform to the vehicle
        transform = random.choice(world.get_map().get_spawn_points())
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        vehicle.set_autopilot(True)

        # "rgb" camera attached to the vehicle.
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        depth_bp = blueprint_library.find('sensor.camera.depth')

        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        depth_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        depth_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '80')
        depth_bp.set_attribute('fov', '80')

        camera_transform = carla.Transform(carla.Location(x=1.5, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        depth = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)
        actor_list.append(depth)
        print('created %s' % depth.type_id)

        image_queue = Queue()
        depth_queue = Queue()

        camera.listen(lambda data: process_img(data, image_queue))
        depth.listen(lambda data: process_img(data, depth_queue))

        # cc = carla.ColorConverter.LogarithmicDepth
        # cc = carla.ColorConverter.Depth
        for frame in range(100):
            world.tick()
            world_frame = world.get_snapshot().frame

            try:
                # Get the data once it's received.
                image_data = image_queue.get(True, 1.0)
                depth_data = depth_queue.get(True, 1.0)
            except Empty:
                print("[Warning] Some sensor data has been missed")
                continue
            
            assert image_data.frame == depth_data.frame == world_frame

            im_array = np.copy(np.frombuffer(image_data.raw_data, dtype=np.dtype("uint8")))
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]

            img = torch.from_numpy(im_array.copy()).float().to(device)


            im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)
            cv2.imshow("window", im_array)
            cv2.waitKey(1)

            cc = carla.ColorConverter.Depth
            depth_data.convert(cc)
            depth_array = np.copy(np.frombuffer(depth_data.raw_data, dtype=np.dtype("uint8")))
            depth_array = np.reshape(depth_array, (depth_data.height, depth_data.width, 4))
            depth_array = depth_array[:, :, :3][:, :, ::-1]

            
            cv2.imshow("output", depth_array)
            cv2.waitKey(1)

        # camera.listen(lambda image: image.save_to_disk('outside/%06d.png' % image.frame))


        

    finally:

        world.apply_settings(original_settings)
        print('destroying actors')
        vehicle.destroy()
        camera.destroy()
        depth.destroy()
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')


if __name__ == '__main__':

    main()
