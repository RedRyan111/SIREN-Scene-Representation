import torch
from transformations.transformations import SpatialTransformationManager


class RaysFromCameraBuilder:
    def __init__(self, image_height, image_width, focal, device):
        self.height = image_height
        self.width = image_width
        self.focal_length = focal
        self.device = device

        self.directions = self.get_ray_directions().to(device)

    def get_ray_directions(self):
        row_meshgrid, col_meshgrid = torch.meshgrid(
            unit_length_torch_arange(self.height, self.focal_length).to(self.device),
            unit_length_torch_arange(self.width, self.focal_length).to(self.device)
        )

        return get_ray_directions_from_meshgrid(row_meshgrid, col_meshgrid)

    def ray_origins_and_directions_from_pose(self, camera_to_world_transform: torch.Tensor):
        cam2world = SpatialTransformationManager(camera_to_world_transform)

        ray_directions = cam2world.rotate_ray_bundle(self.directions)
        ray_origins = cam2world.expand_origin_to_match_ray_bundle_shape(ray_directions)

        return ray_origins, ray_directions


def get_ray_directions_from_meshgrid(row_meshgrid, col_meshgrid):
    directions = torch.stack([
        col_meshgrid,
        -1 * row_meshgrid,
        -torch.ones_like(row_meshgrid)
    ], dim=-1)

    return directions


def unit_length_torch_arange(full_range, resolution):
    bound = .5 * full_range / resolution
    return torch.arange(-1 * bound, bound, 1 / resolution)
