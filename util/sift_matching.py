import os

import cv2
import numpy as np
from matplotlib import pyplot as plt
from DataAugForObjectDetection.xml_helper import parse_xml
import math

ransacReprojThreshold = 4

target_img_dir = "/Users/lichengzhi/bailian/fontdata/result/character"
FLANN_INDEX_KDTREE = 0
FLANN_INDEX_LSH = 6

indexParams = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
searchParams = dict(checks=100)
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
    cmp_list = list()
    source = cv2.imread("/Users/lichengzhi/bailian/fontdata/result/binary.jpg")
    for r, _, files in os.walk(target_img_dir):
        for file in files:
            img = cv2.imread(os.path.join(target_img_dir, file))
            if img is None:
                print("Error target image data: %s\n" % file)
                continue
            target = img.copy()
            label_target = file
            kp_source, des_source = sift_kp(source)
            target = cv2.resize(target, (source.shape[1], source.shape[0]), cv2.INTER_CUBIC)
            kp_target, des_target = sift_kp(target)
            matches = flann.knnMatch(des_source, des_target, k=2)
            match_num, matches_mask = get_match_num(matches, 0.9)
            match_ratio = match_num * 100 / len(matches)
            draw_params = dict(matchColor=(0, 255, 0), singlePointColor=(0, 0, 255), matchesMask=matches_mask)
            cmp_img = cv2.drawMatchesKnn(source, kp_source, target, kp_target, matches, None, **draw_params)
            cmp_list.append((cmp_img, match_ratio, label_target))
            # window_name = label_target + ": " + "%.2f%%" % match_ratio
            # print(window_name)
            # print("\n")
            # cv2.imshow(window_name, cmp_img)
            #
            # cv2.waitKey(0)
            # cv2.destroyWindow(window_name)
    cmp_list.sort(key=lambda x: x[1], reverse=True)
    cmp_img, match_ratio, label_target = cmp_list[0]
    window_name = label_target + ": " + "%.2f%%" % match_ratio
    print(window_name)
    print("\n")
    cv2.imshow(window_name, cmp_img)
    cv2.waitKey(0)
    cv2.destroyWindow(window_name)


if __name__ == "__main__":
    main()
