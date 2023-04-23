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
from torch.autograd import Variable

import carla
import random
import time
import numpy as np
import cv2
from queue import Queue
from queue import Empty
from pixelformer.networks.PixelFormer import PixelFormer
from pixelformer.utils import post_process_depth, flip_lr

IM_WIDTH = 1216
IM_HEIGHT = 352


def process_img(data, queue):

    queue.put(data)



def main():
    actor_list = []
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # device = torch.device("cpu")
    model = PixelFormer(version='large07', inv_depth=False, max_depth=80)
    model = torch.nn.DataParallel(model)
    
    checkpoint = torch.load("examples/PixelFormer/pretrained/kitti.pth")
    model.load_state_dict(checkpoint['model'])
    model.eval()
    model.to(device)

    num_params = sum([np.prod(p.size()) for p in model.parameters()])
    print("Total number of parameters: {}".format(num_params))

    pred_depths = []
    im_arrayss = []
    depth_arrayss = []
    try:
        # requests in the localhost at port 2000.
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()
        original_settings = world.get_settings()

        settings = world.get_settings()
        settings.synchronous_mode = True
        settings.fixed_delta_seconds = 0.05
        world.apply_settings(settings)

        blueprint_library = world.get_blueprint_library()

        # blueprints of type 'vehicle' 
        bp = random.choice(blueprint_library.filter('vehicle'))

        #  initial transform to the vehicle
        # transform = random.choice(world.get_map().get_spawn_points())
        transform = world.get_map().get_spawn_points()[32]
        vehicle = world.spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)

        # Let's put the vehicle to drive around.
        # vehicle.set_autopilot(True)

        # "rgb" camera attached to the vehicle.
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        depth_bp = blueprint_library.find('sensor.camera.depth')

        camera_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        camera_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        depth_bp.set_attribute('image_size_x', f'{IM_WIDTH}')
        depth_bp.set_attribute('image_size_y', f'{IM_HEIGHT}')
        camera_bp.set_attribute('fov', '76')
        depth_bp.set_attribute('fov', '76')

        camera_transform = carla.Transform(carla.Location(x=1.6, z=2.4))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=vehicle)
        depth = world.spawn_actor(depth_bp, camera_transform, attach_to=vehicle)
        actor_list.append(camera)
        print('created %s' % camera.type_id)
        actor_list.append(depth)
        print('created %s' % depth.type_id)

        # SPAWN ALL VEHICLES
        for i in range(len(world.get_map().get_spawn_points())):

            # bp = random.choice(blueprint_library.filter('vehicle')) vehicle.yamaha.yzf  vehicle.carlamotors.firetruck
            bp = blueprint_library.find('vehicle.yamaha.yzf') 

            npc = world.try_spawn_actor(bp, world.get_map().get_spawn_points()[i])
            if npc is not None:
                actor_list.append(npc)
                # print('created %s' % npc.type_id)
                
        image_queue = Queue()
        depth_queue = Queue()

        camera.listen(lambda data: process_img(data, image_queue))
        depth.listen(lambda data: process_img(data, depth_queue))

        # cc = carla.ColorConverter.LogarithmicDepth
        # cc = carla.ColorConverter.Depth
        for frame in range(150):
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
            im_array = cv2.cvtColor(im_array, cv2.COLOR_BGR2RGB)

            cv2.imshow("original", im_array)
            cv2.waitKey(1)
            im_array = im_array[np.newaxis, :]

            cc = carla.ColorConverter.Depth
            # depth_data.convert(cc)
            depth_array = np.copy(np.frombuffer(depth_data.raw_data, dtype=np.dtype("uint8")))
            depth_array = np.reshape(depth_array, (depth_data.height, depth_data.width, 4))
            depth_array = depth_array[:, :, :3][:, :, ::-1]


            cv2.imshow("depth", depth_array)
            cv2.waitKey(1)

            array = depth_array.astype(np.float32)
            normalized_depth = np.dot(array, [1.0, 256.0, 65536.0])
            normalized_depth /= 16777215.0 #  <0, 1> ( * 1000 za vrij u metrima)
            in_meters = normalized_depth * 1000 
            in_meters[in_meters > 80] = 0
            in_meters = in_meters.astype(np.uint8)

            cv2.imshow("in_meters", in_meters)
            cv2.waitKey(1)
            

            with torch.no_grad():
                image = torch.from_numpy(im_array.transpose((0, 3, 1, 2))).to(device)
            
                depth_est = model(image)
                post_process = True
                if post_process:
                    image_flipped = flip_lr(image)
                    depth_est_flipped = model(image_flipped)
                    depth_est = post_process_depth(depth_est, depth_est_flipped)

                pred_depth = depth_est.cpu().detach().numpy().squeeze()
                pred_depths.append(pred_depth)
                
                depth_arrayss.append(depth_array)
                
            # print(f'{pred_depth[260, 626]} m')
            # print("========================")
            
            pred_depth_scaled = pred_depth.astype(np.uint8)
            im_arrayss.append(pred_depth_scaled)
            
            cv2.imshow("predicted", pred_depth_scaled)
            cv2.waitKey(1)
            
            
            
        print(f"Image resolution: {pred_depths[0].shape}")
        print(f"Data type: {pred_depths[0].dtype}")
        print(f"Min value: {np.min(pred_depths[0])}")
        print(f"Max value: {np.max(pred_depths[0])}")
        print("========================")
        
        # print(f"Image resolution: {im_arrayss[0].shape}")
        # print(f"Data type: {im_arrayss[0].dtype}")
        # print(f"Min value: {np.min(im_arrayss[0])}")
        # print(f"Max value: {np.max(im_arrayss[0])}")
        # print("========================")
        
        print(f"Image resolution: {in_meters.shape}")
        print(f"Data type: {in_meters.dtype}")
        print(f"Min value: {np.min(in_meters)}")
        print(f"Max value: {np.max(in_meters)}")
        print("========================")
        print("model depth na (y,x): 200, 620")
        print(f'{pred_depth_scaled[200, 620]} m')

        array = depth_array.astype(np.float32)
        normalized_depth = np.dot(array[200, 620], [1.0, 256.0, 65536.0])
        normalized_depth /= 16777215.0
        in_meters = 1000 * normalized_depth
        
        print("real depth na auto ispred (y,x): 200, 620")
        print(f'{in_meters} m')

        print("========================")
        print("model depth na (y,x): 260, 370")
        print(f'{pred_depth_scaled[260, 370]} m')

        array = depth_array.astype(np.float32)
        normalized_depth = np.dot(array[260, 370], [1.0, 256.0, 65536.0])
        normalized_depth /= 16777215.0
        in_meters = 1000 * normalized_depth
        
        print("real depth na auto u traci do (y,x): 260, 370")
        print(f'{in_meters} m')





        print("========================")
        print("model depth na (y,x): 170, 780")
        print(f'{pred_depth_scaled[170, 780]} m')

        array = depth_array.astype(np.float32)
        normalized_depth = np.dot(array[170, 780], [1.0, 256.0, 65536.0])
        normalized_depth /= 16777215.0
        in_meters = 1000 * normalized_depth
        
        print("real depth na  semfor  do (y,x): 170, 780")
        print(f'{in_meters} m')



        print("========================")
        print("model depth na (y,x): 200, 130")
        print(f'{pred_depth_scaled[200, 130]} m')

        

        print("real depth na stop znak lijevo  do (y,x): 200, 130")
        print(f'{in_meters} m')


        # print(f'{array[300, 78, 2]}, {array[300, 78, 1]}, {array[300, 78, 0]}')

            # world.wait_for_tick()

        # camera.listen(lambda image: image.save_to_disk('outside/%06d.png' % image.frame))

# 1127 232 , 426 269
        

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
