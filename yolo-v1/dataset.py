from torch.utils.data import Dataset, DataLoader
import os
import sys
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms


class TrainDataset(Dataset):
    def __init__(self) -> None:
        super().__init__()
        images_dir = 'data/train_images_448'
        image_paths = []
        root, _, image_names = next(os.walk(images_dir))
        for image_name in image_names:
            image_path = os.path.join(root, image_name)
            image_paths.append(image_path)
        self.image_paths = image_paths

        targets_dir = 'data/targets'
        target_paths = []
        root, _, target_names = next(os.walk(targets_dir))
        for target_name in target_names:
            target_path = os.path.join(root, target_name)
            target_paths.append(target_path)
        self.target_paths = target_paths

        if len(image_paths) != len(target_paths):
            sys.exit("image number not equal to target number")

    def __len__(self):
        return len(self.target_paths)

    def __getitem__(self, idx):
        image = Image.open(self.image_paths[idx])
        image = image.resize((448, 448))
        image_tensor = transforms.ToTensor()(image)

        target = np.load(self.target_paths[idx])
        target = torch.tensor(target, dtype=torch.float)
        return image_tensor, target


if __name__ == '__main__':
    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=64)
    x, y = next(iter(dataloader))
    image = x[0]
    image = image.squeeze().permute(1, 2, 0)
    print(image.dtype, y.dtype)
    plt.figure()
    plt.imshow(image)
    plt.show()
