import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from DataAugForObjectDetection.xml_helper import parse_xml
import math

ransacReprojThreshold = 4

source_img_dir = "/Users/lichengzhi/bailian/CV_ToolBox/work_dir/hx678"
target_img_dir = "/Users/lichengzhi/bailian/CV_ToolBox/work_dir/JPEGImages"
target_xml_dir = "/Users/lichengzhi/bailian/CV_ToolBox/work_dir/Annotations"
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6

indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=200)
flann = cv2.FlannBasedMatcher(indexParams, searchParams)
sift = cv2.xfeatures2d_SIFT.create()


def sift_kp(image):
    kp, des = sift.detectAndCompute(image, None)
    return kp, des


def get_match_num(matches, ratio):
    '''返回特征点匹配数量和匹配掩码'''
    matches_mask = [[0, 0] for i in range(len(matches))]
    match_num = 0
    for i, (m, n) in enumerate(matches):
        if m.distance < ratio * n.distance: #将距离比率小于ratio的匹配点删选出来
            matches_mask[i] = [1, 0]
            match_num += 1
    return match_num, matches_mask


def main():
    base = list()
    cmp_list = list()
    for r, _, files in os.walk(source_img_dir):
        for file in files:
            img = cv2.imread(os.path.join(r, file))
            if img is not None:
                name = file[:-4]
                base.append((img.copy(), name))
    if len(base) is 0:
        print("None base image data\n")
        return -1
    for r, _, files in os.walk(target_img_dir):
        for file in files:
            img = cv2.imread(os.path.join(target_img_dir, file))
            if img is None:
                print("Error target image data: %s\n" % file)
                continue
            xml_name = file[:-4] + ".xml"
            coords = parse_xml(os.path.join(target_xml_dir, xml_name))  # 解析得到box信息，格式为[[x_min,y_min,x_max,y_max,name]]
            labels = [coord[4] for coord in coords]
            coords = [coord[:4] for coord in coords]
            for i in range(0, len(coords)):
                xmin = int(coords[i][0])
                ymin = int(coords[i][1])
                xmax = int(coords[i][2])
                ymax = int(coords[i][3])
                cmp_list.clear()
                label = labels[i]
                source = img[ymin: ymax, xmin: xmax]
                cv2.imshow("source", source)
                cv2.waitKey(0)
                cv2.destroyWindow("source")
                kp_source, des_source = sift_kp(source)
                print(source.shape)
                for target, label_target in base:
                    # target = cv2.resize(target, (source.shape[1], source.shape[0]), cv2.INTER_CUBIC)
                    kp_target, des_target = sift_kp(target)
                    matches = flann.knnMatch(des_source, des_target, k=2)
                    match_num, matches_mask = get_match_num(matches, 0.5)
                    match_ratio = match_num * 100 / len(matches)
                    draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), matchesMask=matches_mask)
                    cmp_img = cv2.drawMatchesKnn(source, kp_source, target, kp_target, matches, None, **draw_params)
                    cmp_list.append((cmp_img, match_ratio, label_target))
                cmp_list.sort(key=lambda x: x[1], reverse=True)
                print("gt label name: %s\n" % label)
                # for cmp_img, match_ratio, label_target in cmp_list:
                cmp_img, match_ratio, label_target = cmp_list[0]
                window_name = label_target + ": " + "%.2f%%" % match_ratio
                print(window_name)
                print("\n")
                cv2.imshow(window_name, cmp_img)
                cv2.waitKey(0)
                cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
