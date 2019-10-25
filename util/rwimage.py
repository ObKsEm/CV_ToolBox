import cv2
import os

source_dir = "/Users/lichengzhi/bailian/壳牌/RoseGoldJPEG"
target_dir = "/Users/lichengzhi/bailian/壳牌/未命名文件夹"


def main():
    num_img = 0
    for r, _, files in os.walk(source_dir):
        for file in files:
            img = cv2.imread(os.path.join(r, file))
            if img is not None:
                print(file)
                name = "RoseGold-%d.jpg" % num_img
                num_img += 1
                cv2.imwrite(os.path.join(target_dir, name), img)


if __name__ == "__main__":
    main()
