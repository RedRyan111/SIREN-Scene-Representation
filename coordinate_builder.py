import torch

from transformations.transformations import SpatialTransformationManager


def camera_ray_to_plucker_ray(camera_to_world_matrix, ray_origin, ray_direction):
    camera_origin = camera_to_world_matrix

    torch.cross(camera_origin, ray_direction)

    return 0


def my_ray_representation(camera_to_world_matrix):
    cam_to_world_obj = SpatialTransformationManager(camera_to_world_matrix)
    origin = cam_to_world_obj.translation

    projection = torch.zeros(3)

    return origin, projection

def project_ray(ray_origin, ray_direction):

    return 0