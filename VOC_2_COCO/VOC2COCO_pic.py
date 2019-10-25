# -*- coding=utf-8 -*-
import os
import shutil
import cv2


def main():

    source_pic_root_path = '/home/lichengzhi/faster-rcnn.pytorch/data/VOCdevkit/SKU110K'
    target_pic_root_path = '/home/lichengzhi/maskrcnn-benchmark/datasets/coco/SKU110K'
    for mode in ['train', 'val', 'test']:
        with open(os.path.join(source_pic_root_path, 'ImageSets/Main', mode + '.txt')) as f:
            im_list = f.readlines()
            for filename in im_list:
                img_name = filename.strip() + '.jpg'
                img = cv2.imread(os.path.join(source_pic_root_path, "JPEGImages", img_name))
                cv2.imwrite(os.path.join(target_pic_root_path, mode, img_name), img)


if __name__ == "__main__":
    main()
