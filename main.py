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

num_iters = 1000
set_random_seeds()
training_config = load_training_config_yaml()
device = get_tensor_device()
data_manager = DataLoader(device)
ray_builder = RaysFromCameraBuilder(data_manager.image_height, data_manager.image_width, data_manager.focal, device)

num_encoding_functions = 0
origin_encoding = PositionalEncoding(num_dim=3, num_encoding_functions=num_encoding_functions)
direction_encoding = PositionalEncoding(num_dim=3, num_encoding_functions=num_encoding_functions)
model_inp_size = 2#(3 + 3 * 2 * num_encoding_functions) + (3 + 3 * 2 * num_encoding_functions)

model = NLFModel().to(device)#num_encoding_functions, num_encoding_functions).to(device)
optim = optim.Adam(model.parameters(), lr=.00001)
MSE_loss = torch.nn.MSELoss()


def generate_pixel_coordinates(device):
    resolution = 100#img.shape[0]
    pixel_linspace = torch.linspace(-1, 1, steps=resolution)
    x, y = torch.meshgrid(pixel_linspace, pixel_linspace)
    pixel_coordinates = torch.cat((x.reshape(-1, 1), y.reshape(-1, 1)), dim=1).to(device)
    return pixel_coordinates

loss_list = []
for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(0)

    cam2world = SpatialTransformationManager(target_tform_cam2world)

    ray_origin, ray_directions = ray_builder.ray_origins_and_directions_from_pose(cam2world)

    # print(f'ray origins: {ray_origin.shape} ray directions: {ray_directions.shape}')

    encoded_ray_origin = origin_encoding.forward(ray_origin)
    #encoded_ray_directions = direction_encoding.forward(ray_directions)

    encoded_ray_directions = generate_pixel_coordinates(device)

    print(f'encoded ray origins: {encoded_ray_origin.shape} encoded ray directions: {encoded_ray_directions.shape}')
    #model_inputs = torch.concatenate((encoded_ray_origin, encoded_ray_directions), dim=-1)#.reshape(-1, model_inp_size)
    model_inputs = encoded_ray_directions

    print(f'min: {torch.min(model_inputs)} max: {torch.max(model_inputs)} mean: {torch.mean(model_inputs)}')

    print(f'model inputs: {model_inputs.shape}')

    print(f'model inputs: {model_inputs.reshape(-1, model_inp_size)[1337:1347]}')
    print(f'model target: {target_img.reshape(-1,3)[1337:1347]}')

    #indecis = torch.nonzero(target_img.reshape(-1, 3))
    #print(f'indecis: {indecis}')
    model.zero_grad()
    optim.zero_grad()

    model_output = model(model_inputs)

    loss = MSE_loss(model_output, target_img.reshape(-1,3))
    loss.backward()

    optim.step()

    loss_list.append(loss.detach().cpu())

    print(f'model output: {model_output.reshape(-1, 3)[1337:1347]}')

    #if (i % 3 == 0 and i < 60) or (i % 50 == 0):
    if i % 50 == 0:
        display_image(i, loss_list, model_output.reshape((100, 100, 3)), target_img)

plt.plot([i for i in range(len(loss_list))], loss_list)
plt.show()
