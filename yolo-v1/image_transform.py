from PIL import Image, ImageOps
import os
from tqdm import tqdm

if __name__ == '__main__':
    print('start image transform')
    output_dir = 'data/train_images_448'
    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    images_dir = 'data/VOCdevkit/VOC2012/JPEGImages'
    root, _, image_names = next(os.walk(images_dir))
    for image_name in tqdm(image_names):
        image_path = os.path.join(root, image_name)
        image = Image.open(image_path)
        image = image.resize((448, 448))

        if image.mode == 'L':
            image = ImageOps.colorize(image, black='black', white='white')
        output_path = os.path.join(output_dir, image_name)
        image.save(output_path)
