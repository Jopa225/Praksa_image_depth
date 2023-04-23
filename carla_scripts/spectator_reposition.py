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

def main():
    actor_list = []

    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(2.0)
        world = client.get_world()

        spectator = world.get_spectator()

        blueprint_library = world.get_blueprint_library()

        # blueprints of type 'vehicle' 
        bp = random.choice(blueprint_library.filter('vehicle'))

        #  initial transform to the vehicle
        transform = world.get_map().get_spawn_points()[32]
        vehicle = world.try_spawn_actor(bp, transform)
        actor_list.append(vehicle)
        print('created %s' % vehicle.type_id)
        world.tick(10.0)
        world_snapshot = world.wait_for_tick()
        actor_snapshot = world_snapshot.find(vehicle.id)
        # spectatooooooooooooooooooooooooooooooooooooooooooooooor
        # location = vehicle.get_location()
        # location.y += 20.0
        spectator.set_transform(actor_snapshot.get_transform())

    finally:
        print('destroying actors')
        if vehicle:
            vehicle.destroy()

        client.apply_batch([carla.command.DestroyActor(x) for x in actor_list])
        print('done.')

if __name__ == '__main__':

    main()
