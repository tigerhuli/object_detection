import cv2
from label_transform import extract_label_from_xml, generate_classes_label_map
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


def horizontal_flip(image, objects):
    new_image = image[:, ::-1, :]
    image_width = int(new_image.shape[1])

    new_objects = objects.copy()
    new_objects[:, [0, 2]] = np.array([image_width, image_width])-new_objects[:, [0, 2]]
    new_objects[:, [0, 2]] = new_objects[:, [2, 0]]

    return new_image, new_objects


def test_horizontal_flip(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = horizontal_flip(image, objects)
    new_objects_info = objectlist2info(new_objects, label2classname)

    image_with_bboxes = detection_box.add_detection_boxes(image, objects_info)
    new_image_with_bboxes = detection_box.add_detection_boxes(new_image, new_objects_info)

    display_effects(image_with_bboxes, new_image_with_bboxes, 'flip')


def resize(image, objects, scale_x, scale_y):
    new_image = cv2.resize(image, None, fx=scale_x, fy=scale_y)
    new_objects = objects.copy()
    new_objects[:, [0, 2]] *= scale_x
    new_objects[:, [1, 3]] *= scale_y

    return new_image, new_objects


def test_resize(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = resize(image, objects, 0.5, 0.5)
    new_objects_info = objectlist2info(new_objects, label2classname)

    image_with_bboxes = detection_box.add_detection_boxes(image, objects_info)
    new_image_with_bboxes = detection_box.add_detection_boxes(new_image, new_objects_info)

    display_effects(image_with_bboxes, new_image_with_bboxes, 'resize')


def objectsinfo2list(objects_info, classname2label):
    objects = np.ndarray((len(objects_info), 5))
    for i, ori_obj in enumerate(objects_info):
        objects[i] = [int(float(ori_obj[0])), int(float(ori_obj[1])), int(float(ori_obj[2])), int(float(ori_obj[3])), int(classname2label[ori_obj[4]])]
    return objects


def objectlist2info(objects, label2classname):
    tuple_list = []
    for obj in objects:
        t = (int(obj[0]), int(obj[1]), int(obj[2]), int(obj[3]), label2classname[obj[4]])
        tuple_list.append(t)
    return tuple_list


def display_effects(image, new_image, name):
    fig, ax = plt.subplots(1, 2)
    fig.suptitle('display image augmentation')

    ax[0].imshow(image[:, :, ::-1])
    ax[0].set_title('original')

    ax[1].imshow(new_image[:, :, ::-1])
    ax[1].set_title(name)
    plt.show()


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

        # test_horizontal_flip(image, info['objects'], classname2label, label2classname)
        test_resize(image, info['objects'], classname2label, label2classname)
