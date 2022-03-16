import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import torch
import numpy as np
import matplotlib.pyplot as plt
from torchvision import transforms
import cv2
import pandas as pd
import sys

sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


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


def targets2objects(targets, image_width, image_height, label_name_map):
    grid_size = int(image_width)/7
    objs = []
    for i in range(targets.shape[0]):
        for j in range(targets.shape[1]):
            if targets[i][j][4] == 0:
                continue
            box = targets[i][j][0:4]
            obj_x, obj_y, obj_width, obj_height = (i+box[0])*grid_size, (j+box[1])*grid_size, box[2]*image_width, box[3]*image_height
            obj_xmin, obj_ymin, obj_xmax, obj_ymax = int(obj_x-obj_width/2), int(obj_y-obj_height/2), int(obj_x+obj_width/2), int(obj_y+obj_height/2)
            one_hot = targets[i][j][10:]
            obj_name = label_name_map[np.argmax(one_hot)]
            obj = (obj_xmin, obj_ymin, obj_xmax, obj_ymax, obj_name)
            objs.append(obj)
    return obj_name, objs


if __name__ == '__main__':
    from tools import detection_box
    dataset = TrainDataset()
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    count = 0
    for x, y in dataloader:
        image = x[0]
        image = image.squeeze().permute(1, 2, 0).numpy()
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        label_name_map_data = pd.read_csv('data/class_label_map.csv')
        label_name_map = {}
        for name, label in zip(label_name_map_data['classes name'], label_name_map_data['label']):
            label_name_map[int(label)] = name
        image_width, image_height = image.shape[0], image.shape[1]
        targets = y[0].numpy()
        obj_name, objs = targets2objects(targets, image_width, image_height, label_name_map)
        if len(objs) == 1:
            continue
        image = detection_box.add_detection_boxes(image, objs)

        cv2.imshow(f'{obj_name}_{count}', image)
        count += 1
        if count > 5:
            break
    cv2.waitKey(0)
