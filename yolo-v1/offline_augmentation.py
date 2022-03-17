from pickletools import uint8
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


def clip_bboxes(image_shape, objects):
    objects[:, 0] = np.maximum(0, objects[:, 0])
    objects[:, 1] = np.maximum(0, objects[:, 1])
    objects[:, 2] = np.minimum(image_shape[1], objects[:, 2])
    objects[:, 3] = np.minimum(image_shape[0], objects[:, 3])

    valid_mask = objects[:, 0] < objects[:, 2]
    objects = objects[valid_mask, :]

    valid_mask = objects[:, 1] < objects[:, 3]
    objects = objects[valid_mask, :]
    return objects


def test_clip_bboxes():
    objects = np.array([[-1, 2, 2, 4, 5], [1, 4, 5, 5, 10], [1, 1, 2, 2, 1]])
    print(f'objects: {objects}')
    objects = clip_bboxes((4, 4, 3), objects)
    print(f'objects: {objects}')


class Resize(object):
    def __init__(self, scale_x, scale_y):
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, image, objects):
        new_image = cv2.resize(image, None, fx=self.scale_x, fy=self.scale_y)
        new_objects = objects.copy()
        new_objects[:, [0, 2]] *= self.scale_x
        new_objects[:, [1, 3]] *= self.scale_y

        xlim, ylim = min(image.shape[1], new_image.shape[1]), min(image.shape[0], new_image.shape[0])
        canvas = np.zeros(image.shape, dtype=np.uint8)
        canvas[:ylim, :xlim, :] = new_image[:ylim, :xlim, :]
        new_image = canvas

        new_objects = clip_bboxes(new_image.shape, new_objects)
        return new_image, new_objects


def test_resize(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = Resize(0.5, 1)(image, objects)
    new_objects_info = objectlist2info(new_objects, label2classname)

    image_with_bboxes = detection_box.add_detection_boxes(image, objects_info)
    new_image_with_bboxes = detection_box.add_detection_boxes(new_image, new_objects_info)

    display_effects(image_with_bboxes, new_image_with_bboxes, 'resize')


class Translate(object):
    def __init__(self, scale_x, scale_y):
        assert scale_x > -1 and scale_x < 1, 'shift_x should in (-1, 1)'
        assert scale_y > -1 and scale_y < 1, 'shift_x should in (-1, 1)'
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, image, objects):
        shift_x, shift_y = int(image.shape[1]*self.scale_x), int(image.shape[0]*self.scale_y)
        new_xmin, new_ymin, new_xmax, new_ymax = max(0, shift_x), max(0, shift_y), min(image.shape[1], image.shape[1]+shift_x), min(image.shape[0], image.shape[0]+shift_y)
        xmin, ymin, xmax, ymax = max(0, -shift_x), max(0, -shift_y), min(image.shape[1], image.shape[1]-shift_x), min(image.shape[0], image.shape[0]-shift_y)
        new_image = np.zeros(image.shape, dtype=np.uint8)
        new_image[new_ymin:new_ymax, new_xmin:new_xmax, :] = image[ymin:ymax, xmin:xmax, :]

        new_objects = objects.copy()
        new_objects[:, :4] = new_objects[:, :4] + [shift_x, shift_y, shift_x, shift_y]
        new_objects = clip_bboxes(new_image.shape, new_objects)
        return new_image, new_objects


def test_translate(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = Translate(-0.2, -0.2)(image, objects)
    new_objects_info = objectlist2info(new_objects, label2classname)

    image_with_bboxes = detection_box.add_detection_boxes(image, objects_info)
    new_image_with_bboxes = detection_box.add_detection_boxes(new_image, new_objects_info)

    display_effects(image_with_bboxes, new_image_with_bboxes, 'translate')


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


def test_augmentation_methods():
    annotations_dir = 'data\VOCdevkit\VOC2012\Annotations'
    image2info, classes_set = extract_label_from_xml(annotations_dir)
    classname2label, label2classname = generate_classes_label_map(classes_set, False)

    images_dir = 'data\VOCdevkit\VOC2012\JPEGImages'
    for image_name, info in image2info.items():
        image_path = os.path.join(images_dir, image_name)
        image = cv2.imread(image_path)

        # test_horizontal_flip(image, info['objects'], classname2label, label2classname)
        # test_resize(image, info['objects'], classname2label, label2classname)
        test_translate(image, info['objects'], classname2label, label2classname)


if __name__ == '__main__':
    from tools import detection_box

    # test_clip_bboxes()
    test_augmentation_methods()
