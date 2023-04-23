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
from queue import Queue
from queue import Empty
import numpy as np
import cv2

def process_img(data, queue):
    queue.put(data)


IM_WIDTH = 1216
IM_HEIGHT = 352

def main():
    actor_list = []

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        original_settings = world.get_settings()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        current_map = world.get_map()
        spawn_transforms = current_map.get_spawn_points()
        

        for spawn_point in spawn_transforms:
            
            location = spawn_point.location
            rotation = spawn_point.rotation
            marker = world.debug.draw_string(location, 'o', draw_shadow=False, color=carla.Color(r=0, g=255, b=0), life_time=40.0, persistent_lines=True)
        print(spawn_transforms[0])
        #my_wapoint = current_map.
        waypoint_list = current_map.generate_waypoints(1.0)
        for waypoint in waypoint_list:
            transform = waypoint.transform
            location = transform.location
            rotation = transform.rotation
            marker = world.debug.draw_string(location, 'o', draw_shadow=False, color=carla.Color(r=255, g=0, b=0), life_time=200.0, persistent_lines=True)

        waypoint_tuple_list = current_map.get_topology()

        blueprint_library = world.get_blueprint_library()

        # blueprints of type 'vehicle' 
        bp = random.choice(blueprint_library.filter('vehicle'))

        #  initial transform to the vehicle
        transform = world.get_map().get_spawn_points()[2]
        vehicle = world.try_spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        camera_bp = blueprint_library.find('sensor.camera.rgb')
        depth_bp = blueprint_library.find('sensor.camera.depth')

        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        depth_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        depth_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '80')
        depth_bp.set_attribute('fov', '80')

        camera_transform = carla.Transform(carla.Location(x=0, z=2.2))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        depth = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)
        actor_list.append(depth)
        print('created %s' % depth.type_id)

        # VISUALIZE CAMERA 
        camera_location = camera.get_location()
        camera_rotation = camera.get_transform().rotation
        marker = world.debug.draw_string(location, 'O', draw_shadow=False, color=carla.Color(r=0, g=0, b=255), life_time=40.0, persistent_lines=True)
        marker = world.debug.draw_point(location, size=5, color=carla.Color(r=0, g=0, b=255), life_time=40.0)
        # SPAWN ALL VEHICLES
        for i in range(len(world.get_map().get_spawn_points())):

            bp = random.choice(blueprint_library.filter('vehicle'))
            npc = world.try_spawn_actor(bp, world.get_map().get_spawn_points()[i])
            if npc is not None:
                actor_list.append(npc)
                # print('created %s' % npc.type_id)

        image_queue = Queue()
        depth_queue = Queue()

        camera.listen(lambda data: process_img(data, image_queue))
        depth.listen(lambda data: process_img(data, depth_queue))

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
            im_array = np.asarray(im_array, dtype=np.float32) / 255.0
            im_array = np.reshape(im_array, (image_data.height, image_data.width, 4))
            im_array = im_array[:, :, :3][:, :, ::-1]
            # print(im_array.mode)
            im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

            cv2.imshow("original", im_array)
            cv2.waitKey(1)

            cc = carla.ColorConverter.Depth
            # depth_data.convert(cc)
            depth_array = np.copy(np.frombuffer(depth_data.raw_data, dtype=np.dtype("uint8")))
            depth_array = np.reshape(depth_array, (depth_data.height, depth_data.width, 4))
            depth_array = depth_array[:, :, :3][:, :, ::-1]

            # 625 267 3 99 nest
            
            cv2.imshow("depth", depth_array)
            cv2.waitKey(1)
        array = depth_array.astype(np.float32)
        normalized_depth = np.dot(array[269, 625, :], [1.0, 256.0, 65536.0])
        normalized_depth /= 16777215.0
        in_meters = 1000 * normalized_depth
        print(f'U metrima: {in_meters}')
        print(f'{array[267, 625, 2]}, {array[267, 625, 1]}, {array[267, 625, 0]}')

        time.sleep(2)

    finally:
        world.apply_settings(original_settings)
        print('destroying actors')
        if vehicle:
            vehicle.destroy()
        if camera:
            camera.destroy()
        if depth:
            depth.destroy()
                
        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')

if __name__ == '__main__':

    main()
