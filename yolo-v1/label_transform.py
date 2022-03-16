import os
from lxml import etree
import pandas as pd
import numpy as np


def extract_label_from_xml():
    file_dir = 'data\VOCdevkit\VOC2012\Annotations'
    root, _, files = next(os.walk(file_dir))
    image2info = {}
    obj_count = 0
    classes_set = set()
    for file in files:
        file_path = os.path.join(root, file)
        content = open(file_path).read()
        data = etree.XML(content)

        width = data.xpath('//annotation/size/width/text()')[0]
        height = data.xpath('//annotation/size/height/text()')[0]
        info = {}
        info['size'] = (width, height)
        objs = []
        name_list = data.xpath('/annotation/object/name/text()')
        xmin_list = data.xpath('/annotation/object/bndbox/xmin/text()')
        ymin_list = data.xpath('/annotation/object/bndbox/ymin/text()')
        xmax_list = data.xpath('/annotation/object/bndbox/xmax/text()')
        ymax_list = data.xpath('/annotation/object/bndbox/ymax/text()')
        for name, xmin, ymin, xmax, ymax in zip(name_list, xmin_list, ymin_list, xmax_list, ymax_list):
            obj = (name, xmin, ymin, xmax, ymax)
            objs.append(obj)
            obj_count += 1
            classes_set.add(name)

        info['objects'] = objs
        image = data.xpath('//annotation/filename/text()')[0]
        image2info[image] = info

    return image2info, classes_set


def generate_classes_label_map(classes):
    classes = list(classes)
    classes.sort()
    labels = range(len(classes))

    classname2label = {}
    for classname, label in zip(classes, labels):
        classname2label[classname] = label

    data = pd.DataFrame({'classes name': classes, 'label': labels})
    data.to_csv('data/class_label_map.csv', index=False)

    return classname2label


def generate_targets(image2info, classname2label):
    S, targets_dir = 7, 'data/targets'
    if not os.path.isdir(targets_dir):
        os.mkdir(targets_dir)

    for image, info in image2info.items():
        image_width = int(info['size'][0])
        image_height = int(info['size'][1])

        grid_width = image_width/S
        grid_height = image_height/S

        targets = np.zeros((S, S, 30))
        objects = info['objects']

        for object in objects:
            xmin = float(object[1])
            ymin = float(object[2])
            xmax = float(object[3])
            ymax = float(object[4])

            x = (xmin+xmax)/2
            y = (ymin+ymax)/2
            w = xmax-xmin
            h = ymax-ymin

            target_x = float(x % grid_width)/grid_width
            target_y = float(y % grid_height)/grid_height
            target_w = float(w)/image_width
            target_h = float(h)/image_height

            cell = np.zeros((30))
            cell[4], cell[9] = 1, 1
            cell[0:4] = [target_x, target_y, target_w, target_h]
            cell[5:9] = [target_x, target_y, target_w, target_h]

            classname = str(object[0])
            cell[10+classname2label[classname]] = 1

            i = int(x/grid_width)
            j = int(y/grid_height)
            targets[i][j] = cell

        np.save(os.path.join(targets_dir, image+'.npy'), targets)


if __name__ == '__main__':
    print('start label transform')
    image2info, classes_set = extract_label_from_xml()
    classname2label = generate_classes_label_map(classes_set)
    generate_targets(image2info, classname2label)
