import cv2

if __name__ == '__main__':
    print('start show detection box')

    image_path = 'data/2007_000027.jpg'
    image = cv2.imread(image_path)
    print(image.shape)
    row_num, col_num = image.shape[0], image.shape[1]
    xmin, ymin, xmax, ymax = 174, 101, 349, 351
    rmin, cmin, rmax, cmax = row_num-ymax, xmin, row_num-ymin, xmax
    object_name = 'person'
    image = cv2.rectangle(image, (xmin, ymin), (xmax, ymax), color=(255, 0, 255), thickness=2)
    label_width, label_height = cv2.getTextSize(object_name + '0', cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]

    image = cv2.rectangle(image, (xmin+label_width, ymin+label_height+3),
                          (xmin, ymin), color=(255, 0, 255), thickness=-1)
    image = cv2.putText(image, object_name, (xmin, ymin+label_height), cv2.FONT_HERSHEY_SIMPLEX,
                        fontScale=0.5, color=(0, 0, 0), thickness=1)

    cv2.imshow('test image', image)  # default color space in cv2 is BGR
    cv2.waitKey(0)
