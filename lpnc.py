"""
This is for clipping stone from an image in a belt
"""
import glob
import os
import numpy as np
import cv2

from utils import read_tif
# get all images from the folders
path = './data/mine'
image_paths = glob.glob(os.path.join(path, '*.tif'))
SHOW_IMAGE = False
for big_I, image_path in enumerate(image_paths):
    # open image
    count = 0
    imarray = read_tif(image_path)
    imarray = cv2.imread(image_path, 2)
    #erode the image
    # cv2.imshow("befor", imarray)
    # cv2.waitKey(0)
    # imarray = cv2.erode(imarray,(3,3),None,iterations = 10)
    # cv2.imshow("after,", imarray)
    # cv2.waitKey(0)
    #dilate the image
    # imarray = cv2.dilate(imarray,(3,3),None,iterations = 2)
    # cut it into HR and IR
    ih = imarray[:, :640]
    il = imarray[:, 640:]

    # use ih to find the stone
    # 1. turn ih into 8-bit image (0~255 for each pixel)
    ih_in_byte = (ih // 255).astype(np.uint8)

    # 2. binarization https://docs.opencv.org/3.4/d4/d73/tutorial_py_contours_begin.html
    ret, thresh = cv2.threshold(ih_in_byte, 160, 255, 0)
    kernel = np.ones((1, 1), np.uint8)
    #cv2.imshow("1", thresh)
    #cv2.waitKey(0)
    # thresh = cv2.dilate(thresh,kernel,iterations = 1)

    # dilate the image
    # thresh = cv2.erode(thresh,kernel,iterations = 1)
    #cv2.imshow("1", thresh)
    #cv2.waitKey(0)
    if SHOW_IMAGE:
        cv2.imshow("threst", thresh)
        cv2.imshow("ih_in_byte", ih_in_byte)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 3. find contours
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    if SHOW_IMAGE:
        cv2.drawContours(ih_in_byte, contours[1:], -1, (0, 0, 255), 2)
        cv2.imshow("img", ih_in_byte)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

   # contours = contours[1:] # remove the outer one which is useless

    # 4. get bounding box using contours https://docs.opencv.org/3.4/da/d0c/tutorial_bounding_rects_circles.html
    contours_poly = [None] * len(contours)
    boundRect = [None] * len(contours)
    for i, c in enumerate(contours):
        contours_poly[i] = cv2.approxPolyDP(c, 3, True)
        boundRect[i] = cv2.boundingRect(contours_poly[i])

    if SHOW_IMAGE:
        for i, c in enumerate(contours):
            cv2.rectangle(ih_in_byte, (int(boundRect[i][0]), int(boundRect[i][1])), \
                         (int(boundRect[i][0] + boundRect[i][2]), int(boundRect[i][1] + boundRect[i][3])), 0, 2)
        cv2.imshow('Contours', ih_in_byte)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    # 5. clip image into small pieces
    stones_ih = []
    stones_il = []
    for x,y,w,h in boundRect:
        stones_ih.append(ih[y:y+h, x:x+w])
        stones_il.append(il[y:y+h, x:x+w])
    pass
    # TODO: save stones into local 1 by 1
    save_ih_path = './data/fine_ih/'
    save_il_path = './data/fine_il/'
    stones_ih = np.array(stones_ih)
    stones_il = np.array(stones_il)

    for i in range(len(contours)):
        area = stones_ih[i].shape[0] * stones_ih[i].shape[1]
        if area > 5000 or area < 20:
            continue

        cv2.imwrite(os.path.join(save_ih_path, "{}_{}.tif".format(big_I,count)),stones_ih[i])
        cv2.imwrite(os.path.join(save_il_path, "{}_{}.tif".format(big_I,count)),stones_il[i])
        count += 1