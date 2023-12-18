import torch


class CameraToWorldSpatialTransformationManager:
    def __init__(self, spatial_matrix):
        self.spatial_matrix = spatial_matrix
        self.orientation = spatial_matrix[:3, :3]
        self.translation = spatial_matrix[:3, -1]
        self.inverse_spacial_matrix = self.build_inverse_spatial_matrix()

    def build_inverse_spatial_matrix(self):
        inverse_spacial_matrix = torch.eye(4)
        inverse_rotation = self.orientation.T
        inverse_translation = -1 * inverse_rotation * self.translation

        inverse_spacial_matrix[:3, :3] = inverse_rotation
        inverse_spacial_matrix[:3, -1] = inverse_translation

        return inverse_spacial_matrix

    def transform_ray_bundle(self, ray_bundle):
        return torch.matmul(ray_bundle, self.orientation.T)

    def expand_origin_to_match_ray_bundle_shape(self, ray_bundle):
        return self.translation.expand(ray_bundle.shape)

# camera to world

# camera to camera
def camera_to_camera(starting_camera_to_world: CameraToWorldSpatialTransformationManager, final_camera_to_world_2: CameraToWorldSpatialTransformationManager):
    starting_inverse_camera_to_world = starting_camera_to_world.inverse_spacial_matrix
    return torch.matmul(starting_inverse_camera_to_world, final_camera_to_world_2.spatial_matrix)


# world to camera