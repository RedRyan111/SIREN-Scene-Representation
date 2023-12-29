import torch.nn

from display_utils.display_utils import display_image
from neural_light_field.models.model import NLFModel
from neural_light_field.positional_encodings import PositionalEncoding
from ray_constructors import RaysFromCameraBuilder
from data_loaders.tiny_data_loader import DataLoader
from tqdm import tqdm
from setup.setup_utils import set_random_seeds, load_training_config_yaml, get_tensor_device
import matplotlib.pyplot as plt
from spatial_transformations.spatial_transformations import SpatialTransformationManager
import torch.optim as optim

num_iters = 100
set_random_seeds()
training_config = load_training_config_yaml()
device = get_tensor_device()
data_manager = DataLoader(device)
ray_builder = RaysFromCameraBuilder(data_manager.image_height, data_manager.image_width, data_manager.focal, device)

num_encoding_functions = 0
origin_encoding = PositionalEncoding(num_dim=3, num_encoding_functions=num_encoding_functions)
direction_encoding = PositionalEncoding(num_dim=3, num_encoding_functions=num_encoding_functions)

model = NLFModel(num_encoding_functions, num_encoding_functions).to(device)
optim = optim.Adam(model.parameters(), lr=.001)
MSE_loss = torch.nn.MSELoss()

loss_list = []
for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(0)

    cam2world = SpatialTransformationManager(target_tform_cam2world)

    ray_origin, ray_directions = ray_builder.ray_origins_and_directions_from_pose(cam2world)

    # print(f'ray origins: {ray_origin.shape} ray directions: {ray_directions.shape}')

    encoded_ray_origin = origin_encoding.forward(ray_origin)
    encoded_ray_directions = direction_encoding.forward(ray_directions)

    print(f'encoded ray origins: {encoded_ray_origin.shape} encoded ray directions: {encoded_ray_directions.shape}')

    model_output = model(encoded_ray_origin, encoded_ray_directions)

    # print(f'model output: {model_output.shape}')

    loss = MSE_loss(model_output, target_img)
    loss.backward()

    optim.step()

    loss_list.append(loss.detach().cpu())

    print(f'model output: {model_output.shape}')

    if i % 10 == 0:
        display_image(i, loss_list, model_output.reshape((100, 100, 3)), target_img)

plt.plot([i for i in range(len(loss_list))], loss_list)
plt.show()
