from neural_light_field.positional_encodings import PositionalEncoding
from ray_constructors import RaysFromCameraBuilder
from data_loaders.tiny_data_loader import DataLoader
from tqdm import tqdm
from setup.setup_utils import set_random_seeds, load_training_config_yaml, get_tensor_device
import matplotlib.pyplot as plt
from spatial_transformations.spatial_transformations import SpatialTransformationManager

num_iters = 10
set_random_seeds()
training_config = load_training_config_yaml()
device = get_tensor_device()
data_manager = DataLoader(device)
ray_builder = RaysFromCameraBuilder(data_manager.image_height, data_manager.image_width, data_manager.focal, device)

origin_encoding = PositionalEncoding(num_dim=3, num_encoding_functions=12)
direction_encoding = PositionalEncoding(num_dim=3, num_encoding_functions=12)

for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(i)

    #print(f'cam to world:')
    #print(target_tform_cam2world)

    cam2world = SpatialTransformationManager(target_tform_cam2world)

    ray_origin, ray_directions = ray_builder.ray_origins_and_directions_from_pose(cam2world)

    print(f'ray origins: {ray_origin.shape} ray directions: {ray_directions.shape}')

    encoded_ray_origin = origin_encoding.forward(ray_origin)
    encoded_ray_directions = direction_encoding.forward(ray_directions)

    print(f'encoded ray origins: {encoded_ray_origin.shape} encoded ray directions: {encoded_ray_directions.shape}')

    #transform rays

    #get plucker coordinates for rays

    #plt.imshow(target_img.cpu())
    #plt.show()
