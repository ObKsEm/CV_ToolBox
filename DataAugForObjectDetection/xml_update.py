import os
import xml.etree.ElementTree as ET
import xml.dom.minidom as DOC

from DataAugForObjectDetection.xml_helper import generate_xml


def main():
    source_xml_path = "/home/lichengzhi/CV_ToolBox/DataAugForObjectDetection/data/Annotations"
    target_xml_path = "/home/lichengzhi/CV_ToolBox/DataAugForObjectDetection/data/Annotations_without_unk"
    for r, dirs, files in os.walk(source_xml_path):
        for file in files:
            xml_path = os.path.join(r, file)
            tree = ET.parse(xml_path)
            root = tree.getroot()
            size = root.find('size')
            width = int(size.find('width').text)
            height = int(size.find('height').text)
            depth = int(size.find('depth').text)
            img_size = (width, height, depth)
            objs = root.findall('object')
            coords = list()
            for ix, obj in enumerate(objs):
                name = obj.find('name').text
                box = obj.find('bndbox')
                x_min = float(box[0].text)
                y_min = float(box[1].text)
                x_max = float(box[2].text)
                y_max = float(box[3].text)
                if name.find("未识别") == -1:
                    coords.append([x_min, y_min, x_max, y_max, name])

            generate_xml(file[:-4] + ".jpg", coords, img_size, target_xml_path)


if __name__ == '__main__':
    main()
