import os
from math import sqrt

import cv2
import numpy as np

input_path = "/Users/lichengzhi/bailian/无人机/白底无人机图片"
output_path = "/Users/lichengzhi/bailian/无人机/clipped"


def edge_clip(img):
    if img is None:
        print("Error image data.")
        return None
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    if img.shape[2] > 3:
        alpha = img[:, :, 3] > 0
        gray[~alpha] = 255
        img[~alpha] = 255
    # cv2.imshow("image", img)
    # cv2.waitKey(0)
    # cv2.destroyWindow("image")
    blurred = cv2.GaussianBlur(gray, (9, 9), 0)
    gradX = cv2.Sobel(blurred, ddepth=cv2.cv2.CV_32F, dx=1, dy=0, ksize=-1)
    gradY = cv2.Sobel(blurred, ddepth=cv2.cv2.CV_32F, dx=0, dy=1, ksize=-1)
    gradient = cv2.subtract(gradX, gradY)
    gradient = cv2.convertScaleAbs(gradient)

    blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
    _, thresh = cv2.threshold(blurred, 40, 255, cv2.THRESH_BINARY)
    # cv2.imshow("thresh", thresh)
    # cv2.waitKey(0)
    # cv2.destroyWindow("thresh")

    kernel_size = min(img.shape[0], img.shape[1]) / 10
    kernel_size = int(kernel_size)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    # cv2.imshow("closed", closed)
    # cv2.waitKey(0)
    # cv2.destroyWindow("closed")
    _, contours, hierarchy = cv2.findContours(closed, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    contours = sorted(contours, key=cv2.contourArea, reverse=True)
    contours = contours[:3]
    center = (img.shape[1] / 2, img.shape[0] / 2)
    distance = np.iinfo(np.int32).max
    sku_contour = contours[0]
    # for contour in contours:
        # mid_x = (contour[:, :, 0].max() + contour[:, :, 0].min()) / 2
        # mid_x = contour[:, :, 0].mean()
        # mid_y = (contour[:, :, 1].max() + contour[:, :, 1].min()) / 2
        # mid_y = contour[:, :, 1].mean()
        # d = sqrt((center[0] - mid_x) ** 2 + (center[1] - mid_y) ** 2)
        # if d < distance:
        #     distance = d
        #     sku_contour = contour

    # empty = np.zeros(img.shape, np.uint8)
    # empty.fill(255)
    # cv2.drawContours(empty, contours, -1, (0, 0, 0), 3)
    # cv2.drawContours(empty, [sku_contour], -1, (0, 0, 255), 3)
    # empty = cv2.resize(empty, (1000, 600), 3)
    # cv2.imshow("counters", empty)
    # cv2.waitKey(0)
    # cv2.destroyWindow("counters")
    rect = cv2.minAreaRect(sku_contour)
    box = np.array(cv2.boxPoints(rect))
    x = [i[0] for i in box]
    y = [i[1] for i in box]
    xmin = max(0, int(min(x)))
    xmax = min(img.shape[1], int(max(x)))
    ymin = max(0, int(min(y)))
    ymax = min(img.shape[0], int(max(y)))
    img_crop = img[ymin: ymax, xmin: xmax]
    return img_crop


def main():
    for r, dirs, files in os.walk(input_path):
        for file in files:
            # file = "/Users/lichengzhi/bailian/无人机/白底无人机图片/mavic pro/22892.jpg"
            img = cv2.imread(os.path.join(r, file), cv2.IMREAD_UNCHANGED)
            if img is not None:
                print(file)
                img_crop = edge_clip(img)
                directory = os.path.join(output_path, r.split('/')[-1])
                if not os.path.exists(directory):
                    os.mkdir(directory)
                name = file
                pos = name.rfind('.')
                name = name[: pos] + ".jpg"
                cv2.imwrite(os.path.join(directory, name), img_crop)
                # cv2.imshow("draw_img", draw_img)
                # cv2.waitKey(0)


if __name__ == "__main__":
    main()
