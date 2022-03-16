import cv2


def add_detection_boxes(image, objects):
    # objects is list of (xmin, ymin, xmax, ymax, object_name)
    for obj in objects:
        xmin, ymin, xmax, ymax, object_name = obj[0], obj[1], obj[2], obj[3], obj[4]
        image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 255), thickness=2)
        (label_width, label_height), label_line_height = cv2.getTextSize(object_name + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, thickness=1)

        if ymin-label_height-label_line_height >= 0:
            image = cv2.rectangle(image, (xmin, ymin), (xmin+label_width, ymin-label_height-label_line_height), color=(255, 0, 255), thickness=-1)
            image = cv2.putText(image, object_name, (xmin, ymin-label_line_height), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)
        else:
            image = cv2.rectangle(image, (xmin, ymin), (xmin+label_width, ymin+label_height+label_line_height), color=(255, 0, 255), thickness=-1)
            image = cv2.putText(image, object_name, (xmin, ymin+label_height), cv2.FONT_HERSHEY_SIMPLEX, fontScale=0.5, color=(0, 0, 0), thickness=1)

    return image

if __name__ == '__main__':
    print('start show detection box')

    image_path = 'data/2007_000027.jpg'
    image = cv2.imread(image_path)
    row_num, col_num = image.shape[0], image.shape[1]
    xmin, ymin, xmax, ymax = 174, 101, 349, 351
    object_name = 'person'

    image = add_detection_boxes(image, [(xmin, ymin, xmax, ymax, object_name)])

    cv2.imshow('test image', image)  # default color space in cv2 is BGR
    cv2.waitKey(0)
