import cv2
import numpy as np
import random

ransacReprojThreshold = random.randint(0, 4)

def sift_kp(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    sift = cv2.xfeatures2d_SIFT.create()
    kp, des = sift.detectAndCompute(image, None)
    kp_image = cv2.drawKeypoints(gray_image, kp, None)
    return kp_image, kp, des


def get_match(des1, des2):
    bf = cv2.BFMatcher()
    matches = bf.knnMatch(des1, des2, k=2)
    ret = []
    for m, n in matches:
        if m.distance < 0.75 * n.distance:
            ret.append(m)

    return ret


def sift_posson_blending(source, target, coords):
    xmin = int(coords[0])
    ymin = int(coords[1])
    xmax = int(coords[2])
    ymax = int(coords[3])
    img2 = target[ymin: ymax, xmin: xmax]
    img1 = cv2.resize(source, (img2.shape[1], img2.shape[0]))
    source_kp_image, source_kp, source_des = sift_kp(img1)
    target_kp_image, target_kp, target_des = sift_kp(img2)
    matches = get_match(source_des, target_des)
    print("find %d matches\n" % len(matches))
    if len(matches) >= 0:
        ptsS = np.float32([source_kp[j.queryIdx].pt for j in matches]).reshape(-1, 1, 2)
        ptsT = np.float32([target_kp[j.trainIdx].pt for j in matches]).reshape(-1, 1, 2)

        H, status = cv2.findHomography(ptsT, ptsS, cv2.RANSAC, ransacReprojThreshold)
        imgOut = cv2.warpPerspective(img1, H, (img2.shape[1], img2.shape[0]), flags=cv2.INTER_LINEAR + cv2.WARP_INVERSE_MAP)

        cv2.imshow("result", imgOut)
        cv2.waitKey(0)
        cv2.destroyWindow("result")

        flatten_mask = np.zeros(target.shape, dtype=target.dtype)
        flatten_mask[ymin: ymax, xmin: xmax] = 255

        mask = 255 * np.ones(imgOut.shape, imgOut.dtype)
        center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
        textureflatten = cv2.illuminationChange(target, flatten_mask, alpha=2, beta=2)
        cv2.imshow("textureflatten", textureflatten)
        cv2.waitKey(0)
        cv2.destroyWindow("textureflatten")
        normal_clone = cv2.seamlessClone(imgOut, textureflatten, mask, center, cv2.NORMAL_CLONE)
        cv2.imshow("normal_clone", normal_clone)
        cv2.waitKey(0)
        cv2.destroyWindow("normal_clone")
        mixed_clone = cv2.seamlessClone(imgOut, target, mask, center, cv2.MIXED_CLONE)
        cv2.imshow("mixed_clone", mixed_clone)
        cv2.waitKey(0)
        # cv2.destroyWindow("mixed_clone")
        return normal_clone, mixed_clone


def force_poisson_blending(source, target, coords):
    xmin = int(coords[0])
    ymin = int(coords[1])
    xmax = int(coords[2])
    ymax = int(coords[3])
    img2 = target[ymin: ymax, xmin: xmax]
    img1 = cv2.resize(source, (img2.shape[1], img2.shape[0]))
    flatten_mask = np.zeros(target.shape, dtype=target.dtype)
    flatten_mask[ymin: ymax, xmin: xmax] = 255
    mask = 255 * np.ones(img1.shape, img1.dtype)
    center = (int((xmin + xmax) / 2), int((ymin + ymax) / 2))
    textureflatten = cv2.illuminationChange(target, flatten_mask, alpha=2, beta=2)
    # cv2.imshow("textureflatten", textureflatten)
    # cv2.waitKey(0)
    # cv2.destroyWindow("textureflatten")
    normal_clone = cv2.seamlessClone(img1, textureflatten, mask, center, cv2.NORMAL_CLONE)
    # cv2.imshow("normal_clone", normal_clone)
    # cv2.waitKey(0)
    # cv2.destroyWindow("normal_clone")
    mixed_clone = cv2.seamlessClone(img1, textureflatten, mask, center, cv2.MIXED_CLONE)
    # cv2.imshow("mixed_clone", mixed_clone)
    # cv2.waitKey(0)
    # cv2.destroyWindow("mixed_clone")
    return normal_clone, mixed_clone


def coordinates_poisson_blending(source, target, center):
    mask = 255 * np.ones(target.shape, dtype=target.dtype)
    # normal_clone = cv2.seamlessClone(target, source, mask, center, cv2.NORMAL_CLONE)
    mixed_clone = cv2.seamlessClone(target, source, mask, center, cv2.MIXED_CLONE)
    return mixed_clone


