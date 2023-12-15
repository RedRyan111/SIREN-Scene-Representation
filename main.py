from coordinate_constructors import RaysFromCameraBuilder, CameraToWorldSpatialTransformationManager
from data_loaders.tiny_data_loader import DataLoader
from tqdm import tqdm
from setup.setup_utils import set_random_seeds, load_training_config_yaml, get_tensor_device
import matplotlib.pyplot as plt

num_iters = 10
set_random_seeds()
training_config = load_training_config_yaml()
device = get_tensor_device()
data_manager = DataLoader(device)
ray_builder = RaysFromCameraBuilder(data_manager.image_height, data_manager.image_width, data_manager.focal, device)

for i in tqdm(range(num_iters)):

    target_img, target_tform_cam2world = data_manager.get_image_and_pose(i)

    cam2world = CameraToWorldSpatialTransformationManager(target_tform_cam2world)

    #transform rays

    #get plucker coordinates for rays

    plt.imshow(target_img.cpu())
    plt.show()
