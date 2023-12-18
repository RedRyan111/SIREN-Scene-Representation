import torch


def camera_ray_to_plucker_ray(camera_to_world_matrix, ray_origin, ray_direction):


    camera_origin = camera_to_world_matrix

    torch.cross(camera_origin, ray_direction)

    return 0