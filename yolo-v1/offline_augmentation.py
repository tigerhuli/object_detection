from pickletools import uint8
import cv2
from label_transform import extract_label_from_xml, generate_classes_label_map
import os
import numpy as np
import sys
import matplotlib.pyplot as plt
sys.path.append(os.path.abspath(os.path.join(os.getcwd(), os.pardir)))


class HorizontalFlip(object):
    def __init__(self):
        pass

    def __call__(self, image, objects):
        new_image = image[:, ::-1, :]
        image_width = int(new_image.shape[1])

        new_objects = objects.copy()
        new_objects[:, [0, 2]] = np.array([image_width, image_width])-new_objects[:, [0, 2]]
        new_objects[:, [0, 2]] = new_objects[:, [2, 0]]

        return new_image, new_objects


def test_horizontal_flip(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = HorizontalFlip()(image, objects)
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


class Scale(object):
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


def test_scale(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = Scale(0.5, 1)(image, objects)
    new_objects_info = objectlist2info(new_objects, label2classname)

    image_with_bboxes = detection_box.add_detection_boxes(image, objects_info)
    new_image_with_bboxes = detection_box.add_detection_boxes(new_image, new_objects_info)

    display_effects(image_with_bboxes, new_image_with_bboxes, 'scale')


class Translate(object):
    def __init__(self, scale_x, scale_y):
        assert scale_x > -1 and scale_x < 1, 'shift_x should in (-1, 1)'
        assert scale_y > -1 and scale_y < 1, 'shift_x should in (-1, 1)'
        self.scale_x = scale_x
        self.scale_y = scale_y

    def __call__(self, image, objects):
        shift_x, shift_y = int(image.shape[1]*self.scale_x), int(image.shape[0]*self.scale_y)
        new_xmin, new_ymin, new_xmax, new_ymax = max(0, shift_x), max(0, shift_y), min(image.shape[1], image.shape[1]+shift_x), min(image.shape[0], image.shape[0]+shift_y)
        # two image just have the opposite movement
        xmin, ymin, xmax, ymax = max(0, -shift_x), max(0, -shift_y), min(image.shape[1], image.shape[1]-shift_x), min(image.shape[0], image.shape[0]-shift_y)
        new_image = np.zeros(image.shape, dtype=np.uint8)
        new_image[new_ymin:new_ymax, new_xmin:new_xmax, :] = image[ymin:ymax, xmin:xmax, :]

        new_objects = objects.copy()
        new_objects[:, :4] = new_objects[:, :4] + [shift_x, shift_y, shift_x, shift_y]
        new_objects = clip_bboxes(new_image.shape, new_objects)
        return new_image, new_objects


def test_translate(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = Translate(-0.2, 0)(image, objects)
    new_objects_info = objectlist2info(new_objects, label2classname)

    image_with_bboxes = detection_box.add_detection_boxes(image, objects_info)
    new_image_with_bboxes = detection_box.add_detection_boxes(new_image, new_objects_info)

    display_effects(image_with_bboxes, new_image_with_bboxes, 'translate')


def rotateobjects(m, objects):
    w = (objects[:, 2]-objects[:, 0]).reshape(-1, 1)
    h = (objects[:, 3]-objects[:, 1]).reshape(-1, 1)

    x1, y1, x4, y4 = objects[:, 0], objects[:, 1], objects[:, 2], objects[:, 3]
    x1 = x1.reshape(-1, 1)
    y1 = y1.reshape(-1, 1)
    x4 = x4.reshape(-1, 1)
    y4 = y4.reshape(-1, 1)

    x2, y2 = x1+w, y1
    x3, y3 = x1, y1+h

    cornors = np.hstack((x1, y1, x2, y2, x3, y3, x4, y4))
    cornors = cornors.reshape(-1, 2)
    cornors = np.hstack((cornors, np.ones((cornors.shape[0], 1), dtype=type(cornors[0][0]))))

    rcornors = np.dot(m, cornors.T).T
    rcornors = rcornors.reshape(-1, 8)

    x_, y_ = rcornors[:, [0, 2, 4, 6]], rcornors[:, [1, 3, 5, 7]]
    xmin, ymin, xmax, ymax = np.min(x_, 1), np.min(y_, 1), np.max(x_, 1), np.max(y_, 1)
    xmin = xmin.reshape(-1, 1)
    ymin = ymin.reshape(-1, 1)
    xmax = xmax.reshape(-1, 1)
    ymax = ymax.reshape(-1, 1)

    return np.hstack((xmin, ymin, xmax, ymax, objects[:, 4].reshape(-1, 1)))


class Rotate(object):
    def __init__(self, angle):
        self.angle = angle

    def __call__(self, image, objects):
        (h, w) = image.shape[:2]
        cx, cy = w//2, h//2

        m = cv2.getRotationMatrix2D((cx, cy), self.angle, 1.0)
        cos, sin = np.abs(m[0, 0]), np.abs(m[0, 1])
        nw = int(w*cos+h*sin)
        nh = int(w*sin+h*cos)
        m[0, 2] += nw/2-cx
        m[1, 2] += nh/2-cy

        image = cv2.warpAffine(image, m, (nw, nh))
        objects = rotateobjects(m, objects)

        image = cv2.resize(image, (w, h))
        scale_x, scale_y = nw/w, nh/h
        objects[:, [0, 2]] /= scale_x
        objects[:, [1, 3]] /= scale_y

        objects = clip_bboxes(image.shape, objects)
        return image, objects


def test_rotate(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = Rotate(10)(image, objects)
    new_objects_info = objectlist2info(new_objects, label2classname)

    image_with_bboxes = detection_box.add_detection_boxes(image, objects_info)
    new_image_with_bboxes = detection_box.add_detection_boxes(new_image, new_objects_info)

    display_effects(image_with_bboxes, new_image_with_bboxes, 'rotate')


class HorizontalShear(object):
    def __init__(self, factor):
        self.factor = factor

    def __call__(self, image, objects):
        m = np.array([[1.0, self.factor, 0.0], [0.0, 1.0, 0.0]])
        (h, w, _) = image.shape
        nw = int(h*self.factor+w)
        image = cv2.warpAffine(image, m, (nw, h))
        image = cv2.resize(image, (w, h))

        objects[:, [0, 2]] += objects[:, [1, 3]]*self.factor
        scale_x = nw/w
        objects[:, [0, 2]] /= scale_x

        objects = clip_bboxes(image.shape, objects)
        return image, objects


def test_horizontalshear(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = HorizontalShear(2)(image, objects)
    new_objects_info = objectlist2info(new_objects, label2classname)

    image_with_bboxes = detection_box.add_detection_boxes(image, objects_info)
    new_image_with_bboxes = detection_box.add_detection_boxes(new_image, new_objects_info)

    display_effects(image_with_bboxes, new_image_with_bboxes, 'shear')


class Sequence(object):
    def __init__(self, augs):
        self.augs = augs

    def __call__(self, image, objects):
        for aug in self.augs:
            image, objects = aug(image, objects)
        return image, objects


def test_sequence(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = Sequence([HorizontalFlip(), Rotate(-10), Scale(0.9, 0.9)])(image, objects)
    new_objects_info = objectlist2info(new_objects, label2classname)

    image_with_bboxes = detection_box.add_detection_boxes(image, objects_info)
    new_image_with_bboxes = detection_box.add_detection_boxes(new_image, new_objects_info)

    display_effects(image_with_bboxes, new_image_with_bboxes, 'sequence')


class ConstantResize(object):
    def __init__(self, tw, th):
        # input target width and height
        self.tw = tw
        self.th = th

    def __call__(self, image, objects):
        (h, w, _) = image.shape
        scale = min(self.tw/w, self.th/h)
        nw, nh = int(round(w*scale)), int(round(h*scale))
        image = cv2.resize(image, (nw, nh))

        canvas = np.zeros((self.th, self.tw, 3), dtype=np.uint8)
        shift_x, shift_y = (self.tw-nw)//2, (self.th-nh)//2
        canvas[shift_y:shift_y+nh, shift_x:shift_x+nw, :] = image
        image = canvas

        objects[:, :4] *= scale
        objects[:, :4] += [shift_x, shift_y, shift_x, shift_y]

        return image, objects


def test_constantresize(image, objects_info, classname2label, label2classname):
    objects = objectsinfo2list(objects_info, classname2label)
    new_image, new_objects = ConstantResize(448, 448)(image, objects)
    new_objects_info = objectlist2info(new_objects, label2classname)

    image_with_bboxes = detection_box.add_detection_boxes(image, objects_info)
    new_image_with_bboxes = detection_box.add_detection_boxes(new_image, new_objects_info)

    display_effects(image_with_bboxes, new_image_with_bboxes, 'constant resize')


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
        # test_translate(image, info['objects'], classname2label, label2classname)
        # test_rotate(image, info['objects'], classname2label, label2classname)
        # test_horizontalshear(image, info['objects'], classname2label, label2classname)
        # test_sequence(image, info['objects'], classname2label, label2classname)
        test_constantresize(image, info['objects'], classname2label, label2classname)


if __name__ == '__main__':
    from tools import detection_box

    # test_clip_bboxes()
    test_augmentation_methods()
