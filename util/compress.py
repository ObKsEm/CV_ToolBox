import cv2
import os
import matplotlib.pyplot as plt

source_dir = "/Users/lichengzhi/bailian/壳牌/RoseGoldJPEG"

if __name__ == "__main__":
    file = "IMG_1692.jpg"
    img = cv2.imread(os.path.join(source_dir, file))
    if img is not None:
        img_out = cv2.resize(img, (1000, 800), interpolation=cv2.INTER_LINEAR)
        # cv2.imshow("compress", img_out)
        # cv2.waitKey(0)
        plt.imshow(img)
        plt.show()
    else:
        print("None input image file.")
