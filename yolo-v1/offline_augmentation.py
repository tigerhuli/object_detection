import cv2
from label_transform import extract_label_from_xml, generate_classes_label_map
import os
import numpy as np
import sys
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


def horizontal_flip(image, objects):
    image = image[:, ::-1, :]
    image_width = int(image.shape[1])
    objects[:, [0, 2]] = np.array([image_width, image_width])-objects[:, [0, 2]]
    objects[:, [0, 2]] = objects[:, [2, 0]]

    return image, objects


def objects_list_to_tuple_list(objects, label2classname):
    tuple_list = []
    for obj in objects:
        t = (int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), label2classname[obj[4]])
        tuple_list.append(t)
    return tuple_list


if __name__ == '__main__':
    from tools import detection_box
    print('start offline augmentation')
    annotations_dir = 'data\VOCdevkit\VOC2012\Annotations'
    image2info, classes_set = extract_label_from_xml(annotations_dir)
    classname2label, label2classname = generate_classes_label_map(classes_set, False)

    images_dir = 'data\VOCdevkit\VOC2012\JPEGImages'
    for image_name, info in image2info.items():
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)

        objects = np.ndarray((len(info['objects']), 5))
        for i, ori_obj in enumerate(info['objects']):
            objects[i] = [int(float(ori_obj[0])), int(float(ori_obj[1])), int(float(ori_obj[2])), int(float(ori_obj[3])), int(classname2label[ori_obj[4]])]

        image_new, _ = horizontal_flip(image, objects)
        objects_tuple = objects_list_to_tuple_list(objects, label2classname)
        image = detection_box.add_detection_boxes(image_new, objects_tuple)
        cv2.imshow(image_name, image)
        break

    cv2.waitKey(0)
