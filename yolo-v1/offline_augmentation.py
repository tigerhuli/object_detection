import cv2
from label_transform import extract_label_from_xml
import os

if __name__ == '__main__':
    print('start offline augmentation')
    annotations_dir = 'data\VOCdevkit\VOC2012\Annotations'
    image2info, _ = extract_label_from_xml(annotations_dir)

    images_dir = 'data\VOCdevkit\VOC2012\JPEGImages'
    for image_name, info in image2info.items():
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)
        cv2.imshow(image_name, image)
        break

    cv2.waitKey(0)
