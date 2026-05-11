
#######
# Visualization utils
#######
from extrassign.ander.config import PLOTS_PATH

import matplotlib.pyplot as plt

def view_image(image, label):
    plt.imshow(image.squeeze(), cmap="gray")
    plt.title(f"Label: {label}")
    plt.axis("off")

    plt.savefig(PLOTS_PATH + "single_number.png")
    plt.show()

def view_images(images, labels, n_cols, n_rows):
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(12, 8))
    axes = axes.flatten()

    for i in range(n_cols * n_rows):
        image, label = images[i].view(28, 28), labels[i]

        axes[i].imshow(image.squeeze(), cmap="gray")
        axes[i].set_title(label)
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(PLOTS_PATH + "multiple_numbers.png")
    plt.show()

def view_reconstructions(images, reconstructions, n_rows, n_cols):
    total_images = min(len(images), n_rows * n_cols)

    fig, axes = plt.subplots(
        n_rows * 2,
        n_cols,
        figsize=(12, 12)
    )
    
    if n_rows * 2 == 1:
        axes = axes.reshape(1, -1)
    elif n_cols == 1:
        axes = axes.reshape(n_rows * 2, 1)

    for i in range(total_images):
        block_row = i // n_cols
        col = i % n_cols

        original_row = block_row * 2
        recon_row = original_row + 1

        # Original
        axes[original_row, col].imshow(
            images[i].squeeze(),
            cmap="gray"
        )
        axes[original_row, col].axis("off")

        # Reconstruction
        axes[recon_row, col].imshow(
            reconstructions[i].squeeze(),
            cmap="gray"
        )
        axes[recon_row, col].axis("off")

    axes[0, 0].set_title("Original")
    axes[1, 0].set_title("Reconstructed")

    plt.tight_layout()
    plt.savefig(PLOTS_PATH + "reconstruction_comp.png")
    plt.show()
