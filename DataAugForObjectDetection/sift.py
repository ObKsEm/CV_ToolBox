import cv2
import numpy as np
np.set_printoptions(threshold=np.inf)

from PIL import Image

img_path = "/Users/lichengzhi/bailian/壳牌/分类2/壳牌产品标签/HX8/壳牌喜力 HX8 全合成润滑油 5W-40 1L.png"


def main():
    # im = Image.open(img_path)
    # im.show()
    img = cv2.imread(img_path, -1)
    # img = cv2.cvtColor(np.asarray(im), cv2.COLOR_RGBA2BGR)
    cv2.imshow("image", img)
    cv2.waitKey(0)
    cv2.destroyWindow("image")
    if img is not None:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)
        cv2.waitKey(0)
        cv2.destroyWindow("gray")
        blurred = cv2.GaussianBlur(gray, (9, 9), 0)
        cv2.imshow("blurred", blurred)
        cv2.waitKey(0)
        cv2.destroyWindow("blurred")
        #
        gradX = cv2.Sobel(blurred, ddepth=cv2.cv2.CV_32F, dx=1, dy=0, ksize=-1)
        gradY = cv2.Sobel(blurred, ddepth=cv2.cv2.CV_32F, dx=0, dy=1, ksize=-1)
        gradient = cv2.subtract(gradX, gradY)
        gradient = cv2.convertScaleAbs(gradient)

        blurred = cv2.GaussianBlur(gradient, (9, 9), 0)
        _, thresh = cv2.threshold(blurred, 90, 255, cv2.THRESH_BINARY)
        cv2.imshow("thresh", thresh)
        # cv2.waitKey(0)
        # cv2.destroyWindow("thresh")
        # kernel_size = math.sqrt(min(img.shape[0], img.shape[1]))
        # kernel_size = int(kernel_size)
        # kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
        # closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        # cv2.imshow("closed", closed)
        # cv2.waitKey(0)
        # cv2.destroyWindow("closed")

        # ret, binary = cv2.threshold(gray, 127, 255, 4)
        binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
        cv2.imshow("binary", binary)
        cv2.waitKey(0)
        cv2.destroyWindow("binary")

        _, contours, hierarchy = cv2.findContours(binary, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

        contours = sorted(contours, key=cv2.contourArea, reverse=True)
        sku_contour = None
        for contour in contours[1:]:
            mid_x = (contour[:, :, 0].max() + contour[:, :, 0].min()) / 2
            if mid_x < float(img.shape[1]) / 7 * 4:
                sku_contour = contour
                break

        empty = np.zeros(img.shape, np.uint8)
        empty.fill(255)
        cv2.drawContours(empty, [sku_contour], -1, (0, 0, 0), 3)
        empty = cv2.resize(empty, (1000, 600), 3)
        cv2.imshow("counters", empty)
        cv2.waitKey(0)

        rect = cv2.minAreaRect(sku_contour)
        box = np.int0(cv2.boxPoints(rect))
        draw_img = cv2.drawContours(img.copy(), [box], -1, (0, 0, 255), 3)
        draw_img = cv2.resize(draw_img, (1000, 600), 3)
        cv2.imshow("draw_img", draw_img)
        cv2.waitKey(0)

        sift = cv2.xfeatures2d.SIFT_create()
        kp = sift.detect(gray, None)  # 找到关键点
        kp_img = cv2.drawKeypoints(gray, kp, img)  # 绘制关键点
        # cv2.imshow('siftkeypoints', kp_img)
        # cv2.waitKey(0)


if __name__ == "__main__":
    main()
