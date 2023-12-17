import torch


class CameraToWorldSpatialTransformationManager:
    def __init__(self, spatial_matrix):
        self.spatial_matrix = spatial_matrix
        self.orientation = spatial_matrix[:3, :3]
        self.translation = spatial_matrix[:3, -1]

    def transform_ray_bundle(self, ray_bundle):
        return torch.matmul(ray_bundle, self.orientation.T)

    def expand_origin_to_match_ray_bundle_shape(self, ray_bundle):
        return self.translation.expand(ray_bundle.shape)

# camera to world

# camera to camera

# world to camera