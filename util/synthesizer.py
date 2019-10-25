import os
from collections import defaultdict

import cv2
import numpy as np
import random

from VOC_2_COCO.xml_helper import generate_xml, parse_xml
from util.poisson_blending import coordinates_poisson_blending

MAX_RANDOM_TIMES = 20


clipped_img_dir = "/Users/lichengzhi/bailian/无人机/clipped"
source_img_dir = "/Users/lichengzhi/bailian/无人机/uav/JPEGImages"
source_xml_dir = "/Users/lichengzhi/bailian/无人机/uav/Annotations"
target_img_dir = "/Users/lichengzhi/bailian/无人机/uav_alter/JPEGImages"
target_xml_dir = "/Users/lichengzhi/bailian/无人机/uav_alter/Annotations"


id2name = {
    0: "精灵3A",
    1: "mavic 2",
    2: "mavic pro",
    3: "spark",
    4: "tello"
}


def check_coordinates(x, y, s_shape, t_shape, coords):
    xmin = x
    ymin = y
    height, width, _ = s_shape
    h, w, _ = t_shape
    xmax = x + w - 1
    ymax = y + h - 1
    if not (5 <= xmin <= xmax <= width - 5 and 5 <= ymin <= ymax <= height - 5):
        return False
    for coord in coords:
        x1 = int(coord[0])
        y1 = int(coord[1])
        x2 = int(coord[2])
        y2 = int(coord[3])
        cxmin = max(xmin, x1)
        cymin = max(ymin, y1)
        cxmax = min(xmax, x2)
        cymax = min(ymax, y2)
        if cxmin < cxmax and cymin < cymax:
            return False
    return True


def main():
    clipped_img = defaultdict(list)
    for r, _, files in os.walk(clipped_img_dir):
        for file in files:
            img = cv2.imread(os.path.join(r, file))
            if img is not None:
                pos = r.rfind("/")
                tag = r[pos + 1:]
                clipped_img[tag].append(img)

    for r, _, files in os.walk(source_img_dir):
        for file in files:
            img = cv2.imread(os.path.join(source_img_dir, file))
            if img is not None:
                print("Synthesising %s\n" % file)
                xml_name = file[:-4] + ".xml"
                source = img.copy()
                coords = parse_xml(os.path.join(source_xml_dir, xml_name))  # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
                labels = [coord[4] for coord in coords]
                coords = [coord[:4] for coord in coords]
                basic_coord = coords[0]
                basic_xmin = int(basic_coord[0])
                basic_ymin = int(basic_coord[1])
                basic_xmax = int(basic_coord[2])
                basic_ymax = int(basic_coord[3])
                basic_width = basic_xmax - basic_xmin
                basic_height = basic_ymax - basic_ymin
                basic_length = max(basic_width, basic_height)
                image_length = max(source.shape[0], source.shape[1])
                cnt = 0
                synthesis_times = random.randint(0, 5)
                while cnt < synthesis_times:
                    tag_num = random.randint(0, 4)
                    tag = id2name[tag_num]
                    length = len(clipped_img[tag])
                    target = clipped_img[tag][random.randint(0, length - 1)]
                    synthesis_length = max(target.shape[0], target.shape[1])
                    basic_low_scale = basic_length / synthesis_length
                    basic_high_scale = min(1, (0.1 * image_length) / synthesis_length)
                    if basic_low_scale < basic_high_scale:
                        scale = random.uniform(basic_low_scale, basic_high_scale)
                    else:
                        scale = basic_low_scale
                    target = cv2.resize(target, dsize=None, fx=scale, fy=scale, interpolation=cv2.INTER_LINEAR)
                    x = random.randint(0, source.shape[1])
                    y = random.randint(0, source.shape[0])
                    for i in range(0, MAX_RANDOM_TIMES):
                        if check_coordinates(x, y, source.shape, target.shape, coords):
                            break
                        x = random.randint(0, source.shape[1])
                        y = random.randint(0, source.shape[0])
                    xmin = x
                    ymin = y
                    xmax = x + target.shape[1] - 1
                    ymax = y + target.shape[0] - 1
                    coords.append([xmin, ymin, xmax, ymax])
                    labels.append("uav")
                    center_x = int((xmin + xmax) / 2)
                    center_y = int((ymin + ymax) / 2)
                    print("source: ", source.shape)
                    print("target: ", target.shape)
                    print("center: (%d, %d)" % (center_x, center_y))
                    mixed_clone = coordinates_poisson_blending(source, target, (center_x, center_y))
                    source = mixed_clone.copy()
                    # window = "%dth synthesize" % (cnt + 1)
                    # cv2.imshow("%dth synthesize", source)
                    # cv2.waitKey(0)
                    # cv2.destroyWindow(window)
                    cnt += 1
                cv2.imwrite(os.path.join(target_img_dir, file), source)
                for i in range(0, len(coords)):
                    coords[i].append(labels[i])
                generate_xml(file, coords, source.shape, target_xml_dir)


if __name__ == "__main__":
    main()
