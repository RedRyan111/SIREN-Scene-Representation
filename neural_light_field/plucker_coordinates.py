import torch


def plucker_coordinates(ray_origin, ray_direction):
    return torch.cross(ray_origin, ray_direction)
