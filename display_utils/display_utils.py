from matplotlib import pyplot as plt


def display_image(iteration, loss_list, rgb_predicted, target_img):
    plt.figure(figsize=(15, 4))
    plt.subplot(131)
    plt.imshow(target_img.detach().cpu().numpy())
    plt.title(f"Target Image")
    plt.subplot(132)
    plt.imshow(rgb_predicted.detach().cpu().numpy())
    plt.title(f"Iteration {iteration}")
    plt.subplot(133)
    plt.plot([i for i in range(len(loss_list))], loss_list)
    plt.title("PSNR")
    plt.show()