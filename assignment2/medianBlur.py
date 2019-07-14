#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
  Finish 2D convolution/filtering by your self. 
#    What you are supposed to do can be described as "median blur", which means by using a sliding window 
#    on an image, your task is not going to do a normal convolution, but to find the median value within 
#    that crop.
#
#    You can assume your input has only one channel. (a.k.a a normal 2D list/vector)
#    And you do need to consider the padding method and size. There are 2 padding ways: REPLICA & ZERO. When 
#    "REPLICA" is given to you, the padded pixels are same with the border pixels. E.g is [1 2 3] is your
#    image, the padded version will be [(...1 1) 1 2 3 (3 3...)] where how many 1 & 3 in the parenthesis 
#    depends on your padding size. When "ZERO", the padded version will be [(...0 0) 1 2 3 (0 0...)]
#
#    Assume your input's size of the image is W x H, kernel size's m x n. You may first complete a version 
#    with O(W·H·m·n log(m·n)) to O(W·H·m·n·m·n)).
#    Follow up 1: Can it be completed in a shorter time complexity?
#
#    Python version:
#    def medianBlur(img, kernel, padding_way):
#        img & kernel is List of List; padding_way a string
#        Please finish your code under this blank
#
#
//   C++ version:
//   void medianBlur(vector<vector<int>>& img, vector<vector<int>> kernel, string padding_way){
//       Please finish your code within this blank  
//   }
"""

import numpy as np
import matplotlib.pyplot as plt
import cv2

##################################################### medianBlur Function ###########################################

def medianBlur(img, kernel=[[1,1,1],[1,1,1],[1,1,1]], padding_way="ZERO"):
    """
    img: Input gray image
    kernel: a List of List
    padding_way: a string, "ZERO" or "REPLICA"
    """

    if len(np.array(img).shape)!=2:
        raise ValueError("Image dimension should be 2")
    if len(np.array(kernel).shape)!=2:
        raise ValueError("Kernel dimension should be 2")

    kernel_shape = np.array(kernel).shape
    paddingx = kernel_shape[0]//2
    paddingy = kernel_shape[1]//2
    #print("paddingx", paddingx, "paddingy", paddingy)
    
    rows, cols= img.shape
    new_img = np.zeros(img.shape)
    img_padding = np.zeros([rows+paddingx*2, cols+paddingy*2])
    img_padding[paddingx:paddingx+rows, paddingy:paddingy+cols] = img
    
    if padding_way == "ZERO":
        pass
    
    elif padding_way == "REPLICA":
        img_padding[:paddingx, paddingy:paddingy+cols] = img[0, :].reshape(1, -1)
        img_padding[-paddingx:, paddingy:paddingy+cols] = img[-1,:].reshape(1, -1)
        img_padding[paddingx:paddingx+rows, :paddingy] = img[:, 0].reshape(-1,1)
        img_padding[paddingx:paddingx+rows, -paddingy:] = img[:, -1].reshape(-1, 1)
    else:
        raise ValueError("Padding type is not applicable")
     
#     print(img_padding.shape)
#     show_img(img_padding, "padding")
 
    for i in range(rows):
        for j in range(cols):
            w = img_padding[i:i+kernel_shape[0], j:j+kernel_shape[1]]
            #print(w.shape, kernel.shape)
            w = np.array(w) * np.array(kernel)
            w = np.sort(w.flatten())
            new_img[i-1, j-1] = w[w.shape[0]//2]
    return new_img.astype(np.uint8)

#####################################################################################################################

#################################################### Use Example ####################################################
"""
Use Example:
1 Image Blurring
2 Image Denoising
"""
# salt papper noise
def salt_papper_noise(img, salt=0.1, papper=0.1):
    img_ = img.copy()
    rows, cols = img_.shape[:2]
    mask_salt = np.random.rand(rows, cols) < salt
    mask_papper = np.random.rand(rows, cols) < papper
    img_[mask_salt] = 255
    img_[mask_papper] = 0
    return img_.astype(np.uint8)   
    
# use matplotlib to show cv2_read_img
def show_img(img, winname='figure1', subplot=False):
    #plt.figure()
    if len(img.shape)==3 and img.shape[2] == 3:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img_rgb)
    else:
        plt.imshow(img, cmap='gray')
    plt.xticks(())
    plt.yticks(())
    plt.title(winname)
    if not subplot:
        plt.show()    

if __name__ == "__main__":
    img = cv2.imread("./img/dog.jpg", 0)
    kernel = np.ones((5,5))
    media_blur = medianBlur(img, kernel)
    img_noise = salt_papper_noise(img)
    denoise = medianBlur(img_noise, kernel)

    plt.figure()
    plt.subplot(2,2,1)
    show_img(img, "Original Image", True)
    plt.subplot(2,2,2)
    show_img(media_blur, "Media_Blur", True)
    plt.subplot(2,2,3)
    show_img(img_noise, "Image with salt_papper", True)
    plt.subplot(2,2,4)
    show_img(denoise, "Median Denoising", True)
    plt.show()
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
