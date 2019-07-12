#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np
from matplotlib import pyplot as plt

####################################### 图像基本操作（读取、显示、存储等）###########################################
"""
cv2.imread('img_path', flags) 读取图片
flags:指定以何种方式加载图片，有三个取值： 
cv2.IMREAD_COLOR:读取一副彩色图片，图片的透明度会被忽略，默认为该值，实际取值为1；
cv2.IMREAD_GRAYSCALE:以灰度模式读取一张图片，实际取值为0
cv2.IMREAD_UNCHANGED:加载一副彩色图像，透明度不会被忽略。

cv2.imshow('winname', img) 显示图片，窗口自动适应图片大小，img为图片矩阵（numpy.ndarray）

cv2.waitKey() 让程序挂起，等待指定时间的键盘事件，参数为0表示等待无限长的事件
cv2.destroyAllWindows() 销毁所有已创建窗口 
cv2.destroyWindow() 销毁指定窗口，接受一个窗口名字
cv2.namedWindow() 放大缩小窗口

cv2.imwrite('filename', img)
"""
img = cv2.imread('./img/dog.jpg', cv2.IMREAD_COLOR)
cv2.imshow('dog', img)
key = cv2.waitKey()
if key == 27:    # 按下Esc时退出
    cv2.destroyAllWindows()
elif key == ord('s'):  # 按下s时保存
    cv2.imwrite('./img/dog1.jpg', img)
    cv2.destroyAllWindows()

print(img.dtype)
print(img.shape)

# use matplotlib to show image：
# show gray image
img_gray = cv2.imread('./img/dog.jpg', 0)
plt.imshow(img_gray, cmap='gray', interpolation = 'bicubic')
plt.title('gray')
plt.show()

# show color image
"""
opencv以BGR模式加载图片，而matplotlib以RGB模式显示图片，所以用opencv加载的彩色图片，在matplotlib中不能正确显示
"""
img_bgr = cv2.imread('./img/dog.jpg', 1)
img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
# b, g, r = cv2.split(img_bgr)
# img_rgb = cv2.merge([r, g, b])

plt.subplot(1,2,1)
plt.imshow(img_bgr)
plt.title('BGR')
plt.subplot(1,2,2)
plt.imshow(img_rgb)
plt.title('RGB')
plt.show()

"""
C++ 语法：
import cv2 -> include<opencv2/core/core2.hpp>
cv2.imread() -> cv2::imread()
cv2.imshow() -> cv2.imshow()
cv2.imwrite() -> cv2::imwrite()
"""