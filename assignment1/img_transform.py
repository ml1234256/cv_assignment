#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Please combine image crop, color shift, rotation and perspective transform together to complete a data augmentation script.
Your code need to be completed in Python/C++ in .py or .cpp file with comments and readme file to indicate how to use.
"""
import numpy as np
import math
import random
import cv2

def img_transform(img, crop_size=None, crop_type='center', color_shift=False, rotation=False, perspective=False):
    """
    img: imput image
    crop_size: crop_size,[height, width],int
    crop_type: 'center' crop or 'random' crop
    color shift: if change color, bool
    rotation: if rotation, bool
    perspective: if perspective transform, bool
    """
    height, width, channel = img.shape
    if crop_size:
        if crop_type == 'center':
            r = math.floor(height/2) - math.floor(crop_size[0]/2)
            c = math.floor(width/2) - math.floor(crop_size[1]/2)
            img = img[r:(r+crop_size[0]), c:(c+crop_size[1])]
        elif crop_type == 'random':
            rand_x = random.randint(0, max(0, height - crop_size[0] -1))
            rand_y = random.randint(0, max(0, width - crop_size[1]-1))
            img = img[rand_x:(rand_x+crop_size[0]), rand_y:(rand_y+crop_size[1])]
        else:
            raise ValueError('no crop type')

    if color_shift:
        b, g, r = cv2.split(img)
        rang = 50
        for ch in [b, g, r]:
            rand = random.randint(-rang, rang)
            if rand == 0:
                pass
            elif rand > 0:
                lim = 255 - rand
                ch[ch > lim] = 255
                ch[ch <= lim] = (rand + ch[ch <= lim]).astype(img.dtype)
            elif rand < 0:
                lim = 0 - rand
                ch[ch < lim] = 0
                ch[ch >= lim] = (rand + ch[ch >= lim]).astype(img.dtype)
        img_merge = cv2.merge([b, g, r])
        img = img_merge

    if rotation:
        rotation_angle = random.uniform(0, 360)
        mat = cv2.getRotationMatrix2D((img.shape[1]/2, img.shape[0]/2), rotation_angle, 1)
        img = cv2.warpAffine(img, mat, (int(img.shape[1]*1.3), int(img.shape[0]*1.3)))

    if perspective:
        random_margin = 60
        x1 = random.randint(-random_margin, random_margin)
        y1 = random.randint(-random_margin, random_margin)
        x2 = random.randint(width - random_margin - 1, width - 1)
        y2 = random.randint(-random_margin, random_margin)
        x3 = random.randint(width - random_margin - 1, width - 1)
        y3 = random.randint(height - random_margin - 1, height - 1)
        x4 = random.randint(-random_margin, random_margin)
        y4 = random.randint(height - random_margin - 1, height - 1)

        dx1 = random.randint(-random_margin, random_margin)
        dy1 = random.randint(-random_margin, random_margin)
        dx2 = random.randint(width - random_margin - 1, width - 1)
        dy2 = random.randint(-random_margin, random_margin)
        dx3 = random.randint(width - random_margin - 1, width - 1)
        dy3 = random.randint(height - random_margin - 1, height - 1)
        dx4 = random.randint(-random_margin, random_margin)
        dy4 = random.randint(height - random_margin - 1, height - 1)

        pts1 = np.float32([[x1, y1], [x2, y2], [x3, y3], [x4, y4]])
        pts2 = np.float32([[dx1, dy1], [dx2, dy2], [dx3, dy3], [dx4, dy4]])
        M_warp = cv2.getPerspectiveTransform(pts1, pts2)
        img_warp = cv2.warpPerspective(img, M_warp, (width, height))
        img = img_warp
    return img


# img = cv2.imread('./img/dog.jpg', 1)
#
# img = img_transform(img, crop_size=[200,150], crop_type='random', color_shift=True, rotation=True, perspective=True)
#
# cv2.imshow('',img)
# key = cv2.waitKey()
# if key == 27:
#     cv2.destroyAllWindows()