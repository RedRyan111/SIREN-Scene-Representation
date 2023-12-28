import torch


class SpatialTransformationManager:
    def __init__(self, spatial_matrix):
        #self.camera_to_world = spatial_matrix
        # self.world_to_camera = self.build_inverse_spatial_matrix(spatial_matrix)

        self.spatial_matrix = spatial_matrix
        self.orientation = spatial_matrix[:3, :3]
        self.translation = spatial_matrix[:3, -1]

    def rotate_ray_bundle(self, ray_bundle):
        return torch.matmul(ray_bundle, self.orientation.T)

    def expand_origin_to_match_ray_bundle_shape(self, ray_bundle):
        return self.translation.expand(ray_bundle.shape)


def inverse_spatial_matrix(spatial_matrix):
    inverse_spacial_matrix = torch.eye(4)
    inverse_rotation = spatial_matrix[:3, :3].T
    inverse_translation = (-1 * inverse_rotation) @ spatial_matrix[:3, 3]

    inverse_spacial_matrix[:3, :3] = inverse_rotation
    inverse_spacial_matrix[:3, 3] = inverse_translation

    return inverse_spacial_matrix


# camera to camera
def camera_to_camera(starting_camera_to_world: SpatialTransformationManager,
                     final_camera_to_world: SpatialTransformationManager):

    starting_camera_to_world_matrix = starting_camera_to_world.spatial_matrix
    final_world_to_camera_matrix = inverse_spatial_matrix(final_camera_to_world.spatial_matrix)
    return torch.matmul(starting_camera_to_world_matrix, final_world_to_camera_matrix)

# world to camera
