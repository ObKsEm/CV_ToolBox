# -*- coding=utf-8 -*-
import os
import shutil
import cv2


def main():

    source_pic_root_path = '/home/lichengzhi/mmdetection/data/VOCdevkit/shell/2020.05.26'
    target_pic_root_path = '/home/lichengzhi/mmdetection/data/coco/shell/2020.05.26'
    for mode in ['train', 'val', 'test']:
        with open(os.path.join(source_pic_root_path, 'ImageSets/Main', mode + '.txt')) as f:
            im_list = f.readlines()
            if not os.path.exists(os.path.join(target_pic_root_path, mode)):
                os.makedirs(os.path.join(target_pic_root_path, mode))
            for filename in im_list:
                img_name = filename.strip() + '.jpg'
                # img = cv2.imread(os.path.join(source_pic_root_path, "JPEGImages", img_name))
                # cv2.imwrite(os.path.join(target_pic_root_path, mode, img_name), img)
                shutil.copy(os.path.join(source_pic_root_path, "JPEGImages", img_name), os.path.join(target_pic_root_path, mode))


if __name__ == "__main__":
    main()
