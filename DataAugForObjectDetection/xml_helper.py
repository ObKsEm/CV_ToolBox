# -*- coding=utf-8 -*-
import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOC


# 从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
def parse_xml(xml_path):
    '''
    输入：
        xml_path: xml的文件路径
    输出：
        从xml文件中提取bounding box信息, 格式为[[x_min, y_min, x_max, y_max, name]]
    '''
    tree = ET.parse(xml_path)		
    root = tree.getroot()
    objs = root.findall('object')
    coords = list()
    for ix, obj in enumerate(objs):
        name = obj.find('name').text
        box = obj.find('bndbox')
        x_min = float(box[0].text)
        y_min = float(box[1].text)
        x_max = float(box[2].text)
        y_max = float(box[3].text)
        coords.append([x_min, y_min, x_max, y_max, name])
    return coords


# 将bounding box信息写入xml文件中, bouding box格式为[[x_min, y_min, x_max, y_max, name]]
def generate_xml(img_name, coords, img_size, out_root_path):
    '''
    输入：
        img_name：图片名称，如a.jpg
        coords:坐标list，格式为[[x_min, y_min, x_max, y_max, name]]，name为概况的标注
        img_size：图像的大小,格式为[h,w,c]
        out_root_path: xml文件输出的根路径
    '''
    doc = DOC.Document()  # 创建DOM文档对象

    annotation = doc.createElement('annotation')
    doc.appendChild(annotation)

    filename = doc.createElement('filename')
    filename_text = doc.createTextNode(img_name)
    filename.appendChild(filename_text)
    annotation.appendChild(filename)

    path = doc.createElement('path')
    path_text = doc.createTextNode('/近景-油品/' + img_name[:-4])
    path.appendChild(path_text)
    annotation.appendChild(path)

    source = doc.createElement('source')
    annotation.appendChild(source)

    database = doc.createElement('database')
    database_text = doc.createTextNode('Unknown')
    database.appendChild(database_text)
    source.appendChild(database)

    amount = doc.createElement('amount')
    amount_text = doc.createTextNode(str(len(coords)))
    amount.appendChild(amount_text)
    annotation.appendChild(amount)

    size = doc.createElement('size')
    width = doc.createElement('width')
    width_text = doc.createTextNode(str(img_size[1]))
    width.appendChild(width_text)
    size.appendChild(width)

    height = doc.createElement('height')
    height_text = doc.createTextNode(str(img_size[0]))
    height.appendChild(height_text)
    size.appendChild(height)

    depth = doc.createElement('depth')
    depth_text = doc.createTextNode(str(img_size[2]))
    depth.appendChild(depth_text)
    size.appendChild(depth)

    annotation.appendChild(size)

    segmented = doc.createElement('segmented')
    segmented.appendChild(doc.createTextNode('0'))
    annotation.appendChild(segmented)

    for coord in coords:

        object = doc.createElement('object')
        name = doc.createElement('name')
        name_text = doc.createTextNode(coord[4])
        name.appendChild(name_text)
        object.appendChild(name)

        pose = doc.createElement('pose')
        pose.appendChild(doc.createTextNode('Unspecified'))
        object.appendChild(pose)

        xmltype = doc.createElement('type')
        xmltype.appendChild(doc.createTextNode('rect'))
        object.appendChild(xmltype)

        truncated = doc.createElement('truncated')
        truncated.appendChild(doc.createTextNode('1'))
        object.appendChild(truncated)

        difficult = doc.createElement('difficult')
        difficult.appendChild(doc.createTextNode('0'))
        object.appendChild(difficult)

        bndbox = doc.createElement('bndbox')
        xmin = doc.createElement('xmin')
        xmin_text = doc.createTextNode(str(float(coord[0])))
        xmin.appendChild(xmin_text)
        bndbox.appendChild(xmin)
        ymin = doc.createElement('ymin')
        ymin_text = doc.createTextNode(str(float(coord[1])))
        ymin.appendChild(ymin_text)
        bndbox.appendChild(ymin)
        xmax = doc.createElement('xmax')
        xmax_text = doc.createTextNode(str(float(coord[2])))
        xmax.appendChild(xmax_text)
        bndbox.appendChild(xmax)
        ymax = doc.createElement('ymax')
        ymax_text = doc.createTextNode(str(float(coord[3])))
        ymax.appendChild(ymax_text)
        bndbox.appendChild(ymax)
        object.appendChild(bndbox)

        width = doc.createElement('width')
        width.appendChild(doc.createTextNode(str(float(coord[2]) - float(coord[0]))))
        object.appendChild(width)
        height = doc.createElement('height')
        height.appendChild(doc.createTextNode(str(float(coord[3]) - float(coord[1]))))
        object.appendChild(height)

        annotation.appendChild(object)

    # 将DOM对象doc写入文件
    f = open(os.path.join(out_root_path, img_name[:-4]+'.xml'), 'w')
    f.write(doc.toprettyxml(indent='\t'))
    f.close()
