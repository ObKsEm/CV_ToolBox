# -*- coding=utf-8 -*-
import json
import os
import cv2
import xml.etree.ElementTree as ET
import shutil


# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    tree = ET.parse(xml_path)
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = int(float(box[0].text))
        y_min = int(float(box[1].text))
        x_max = int(float(box[2].text))
        y_max = int(float(box[3].text))
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords


def convert(root_path, target_json_root_path):
    """
    root_path:
        根路径，里面包含JPEGImages(图片文件夹)，classes.txt(类别标签),以及annotations文件夹(如果没有则会自动创建，用于保存最后的json)
    source_xml_root_path:
        VOC xml文件存放的根目录
    target_xml_root_path:
        coco xml存放的根目录
    phase:
        状态：'train'或者'test'
    split:
        train和test图片的分界点数目

    """

    source_xml_root_path = os.path.join(root_path, "Annotations")
    # 打开类别标签
    # with open(os.path.join(root_path, 'classes.txt')) as f:
    #     classes = f.read().strip().split()
    classes = (
        'FB_ICON_000000',
        'FB_ICON_000001',
        'FB_ICON_000002',
        'FB_ICON_000003',
        'FB_ICON_000004',
        'FB_ICON_000005',
        'FB_ICON_000006',
        'FB_ICON_000007',
        'FB_ICON_000008',
        'FB_ICON_000009',
        'FB_ICON_000011',
        'FB_ICON_000012',
        'FB_ICON_000013',
        'FB_ICON_000014',
        'FB_ICON_000015',
        'FB_ICON_000016',
        'FB_ICON_000017',
        'FB_ICON_000018',
        'FB_ICON_000022',
        'FB_ICON_000024',
        'FB_ICON_000025',
        'FB_ICON_000026',
        'FB_ICON_000027',
        'FB_ICON_000028',
        'FB_ICON_000029',
        'FB_ICON_000030',
        'FB_ICON_000031',
        'FB_ICON_000032',
        'FB_ICON_000033',
        'FB_ICON_000034',
        'FB_ICON_000035',
        'FB_ICON_000036',
        'FB_ICON_000043',
        'FB_ICON_000044',
        'FB_ICON_000045',
        'FB_ICON_000046',
        'FB_ICON_000047',
        'FB_ICON_000048',
        'FB_ICON_000049',
        'FB_ICON_000050',
        'FB_ICON_000051',
        'FB_ICON_000052',
        'FB_ICON_000053',
        'FB_ICON_000054',
        'FB_ICON_000055',
        'FB_ICON_000056',
        'FB_ICON_000057',
        'FB_ICON_000058',
        'FB_ICON_000059',
        'FB_ICON_000060',
        'FB_ICON_000061',
        'FB_ICON_000062',
        'FB_ICON_000063',
        'FB_ICON_000064',
        'FB_ICON_000065'
    )
    # 生成类别标签
    # classes = {
    #     'KV板-法拉利70年创新合作伙伴',
    # }
    # classes = (
    #     '壳牌恒护超凡喜力系列 1L',
    #     '壳牌恒护超凡喜力系列 4L',
    #     '壳牌先锋超凡喜力系列 4L',
    #     '壳牌先锋超凡喜力系列 1L',
    #     '壳牌超凡喜力系列 1L',
    #     '壳牌超凡喜力系列 4L',
    #     '壳牌极净超凡喜力系列 1L',
    #     '壳牌极净超凡喜力系列 4L',
    #     '其他'
    # )
    categories = {}
    for i, cls in enumerate(classes, 0):
        categories[cls] = i  # mark

    print('---------------- generate categories ---------------')
    # cats = set()
    # for mode in ['train', 'val', 'test']:
    #     im_set_dir = os.path.join(root_path, 'ImageSets/Main')
    #     with open(os.path.join(im_set_dir, mode + '.txt')) as f:
    #         im_list = f.readlines()
    #     pics = [filename.strip() + ".jpg" for filename in im_list]
    #     for i, pic in enumerate(pics):
    #         xml_path = os.path.join(source_xml_root_path, pic[:-4] + '.xml')
    #         try:
    #             coords = parse_xml(xml_path)
    #         except:
    #             print(pic[:-4] + '.xml not exists~')
    #             continue
    #         for coord in coords:
    #             category = coord[4]
    #             cats.add(category)
    # cats = list(cats)
    # cats.sort()
    # ids = range(1, len(cats) + 1)  # 从1开始，0表示背景
    # categories = dict(zip(cats, ids))
    # categories = {'character': 1}
    # print(categories)

    # 读取images文件夹的图片名称
    for mode in ['train', 'val', 'test']:

        dataset = {'categories': [], 'images': [], 'annotations': []}
        # 建立类别标签和数字id的对应关系
        # for i, cls in enumerate(classes, 1):
        #     dataset['categories'].append({'id': i, 'name': cls, 'supercategory': 'none'})  # mark
        im_set_dir = os.path.join(root_path, 'ImageSets/Main')
        with open(os.path.join(im_set_dir, mode + '.txt')) as f:
            im_list = f.readlines()
        pics = [filename.strip() + ".jpg" for filename in im_list]
        print('---------------- start convert ---------------')
        bnd_id = 1  # 初始为1
        for i, pic in enumerate(pics):
            # print('pic  '+str(i+1)+'/'+str(len(pics)))
            xml_path = os.path.join(source_xml_root_path, pic[:-4] + '.xml')
            pic_path = os.path.join(root_path, 'JPEGImages/' + pic)
            # 用opencv读取图片，得到图像的宽和高
            im = cv2.imread(pic_path)
            height, width, _ = im.shape
            # 添加图像的信息到dataset中
            dataset['images'].append({'file_name': pic,
                                      'id': i,
                                      'width': width,
                                      'height': height})
            try:
                coords = parse_xml(xml_path)
            except:
                print(pic[:-4] + '.xml not exists~')
                continue
            for coord in coords:
                # x_min
                x1 = int(coord[0]) - 1
                x1 = max(x1, 0)
                # y_min
                y1 = int(coord[1]) - 1
                y1 = max(y1, 0)
                # x_max
                x2 = int(coord[2])
                # y_max
                y2 = int(coord[3])
                if x1 > x2:
                    k = x1
                    x1 = x2
                    x2 = k
                if y1 > y2:
                    k = y1
                    y1 = y2
                    y2 = k
                # name
                name = coord[4]
                # cls_id = classes.index(name) + 1  # 从1开始
                category = name
                # category = "character"
                category_id = categories[category]
                width = max(0, x2 - x1)
                height = max(0, y2 - y1)
                dataset['annotations'].append({
                    'area': width * height,
                    'bbox': [x1, y1, width, height],
                    'category_id': category_id,
                    'id': bnd_id,
                    'image_id': i,
                    'iscrowd': 0,
                    # mask, 矩形是从左上角点按顺时针的四个顶点
                    'segmentation': [[x1, y1, x2, y1, x2, y2, x1, y2]]
                })
                bnd_id += 1
        # 保存结果的文件夹
        for cate, cid in categories.items():
            cat = {'supercategory': 'none', 'id': cid, 'name': cate}
            dataset['categories'].append(cat)
        folder = os.path.join(target_json_root_path, 'annotations')
        if not os.path.exists(folder):
            os.makedirs(folder)
        json_name = os.path.join(target_json_root_path, 'annotations/instances_{}.json'.format(mode))
        with open(json_name, 'w') as f_json:
            json.dump(dataset, f_json)


if __name__ == '__main__':
    convert(root_path='/home/lichengzhi/mmdetection/data/VOCdevkit/rzx/2020.04.17',
            target_json_root_path='/home/lichengzhi/detectron2/data/rzx')
