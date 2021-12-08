import torch
import torch.optim as optim
import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from utils import (
    gradient_penalty,
    plot_to_tensorboard,
    save_checkpoint,
    load_checkpoint,
    generate_examples,
)
from model import Discriminator, Generator
from math import log2
from tqdm import tqdm
import config
import cv2
import numpy as np
from scipy.stats import truncnorm
from torchvision.utils import save_image

torch.backends.cudnn.benchmarks = False


def cv_show(img):
    cv2.imshow("res", img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def main(n):
    # init
    gen = Generator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)
    critic = Discriminator(
        config.Z_DIM, config.IN_CHANNELS, img_channels=config.CHANNELS_IMG
    ).to(config.DEVICE)

    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99))
    opt_critic = optim.Adam(
        critic.parameters(), lr=config.LEARNING_RATE, betas=(0.0, 0.99)
    )

    # load model
    load_checkpoint(
        config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE,
    )
    load_checkpoint(
        config.CHECKPOINT_CRITIC, critic, opt_critic, config.LEARNING_RATE,
    )

    gen.eval()
    critic.eval()

    alpha = 1.0
    steps = int(log2(256 / 4))

    truncation = 0.7
    for i in range(n):
        with torch.no_grad():
            noise = torch.tensor(truncnorm.rvs(-truncation, truncation, size=(1, config.Z_DIM, 1, 1)),
                                 device=config.DEVICE, dtype=torch.float32)
            img = gen(noise, alpha, steps)
            save_image(img * 0.5 + 0.5, f"saved_256/img_{i}.png")


if __name__ == '__main__':
    main(100)  # num of results you want

