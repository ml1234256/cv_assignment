#-*- coding: UTF-8 -*-
import sys
from PyQt5.QtWidgets import *
from PyQt5.QtGui import QPixmap, QImage
from PyQt5.QtCore import Qt
import torch
from torch.autograd import Variable
import deblur
from PIL import Image
import numpy as np


class Deblur_UI(QTabWidget):
    def __init__(self):
        super(Deblur_UI, self).__init__()
        self.initUI()
        self.blurredImagePath = ""
        self.blurredImage = np.array(())
        self.sharpImage = np.array(())
        self.showImage = np.array(())
        self.openSize = []
        self.showSize = []
        self.scale = 1

    def initUI(self):

        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenHeight = self.screenRect.height()
        self.screenWidth = self.screenRect.width()
        #print(self.screenHeight, self.screenWidth)

        self.winSize = [int(self.screenWidth/2), self.screenHeight - 100]

        self.blurredImageOpen = QPushButton("打开")
        self.blurredImageOpen.setFixedSize(45, 28)
        self.blurredImageOpen.setStyleSheet("border: 0px")

        self.saveImage = QPushButton("保存")
        self.saveImage.setFixedSize(45, 28)
        self.saveImage.setStyleSheet("border: 0px")

        self.deblurImage = QPushButton("去模糊")
        self.deblurImage.setFixedSize(60, 28)
        self.deblurImage.setStyleSheet("border: 0px")

        self.showImagePre = QPushButton("显示原图")
        self.showImagePre.setFixedSize(70, 28)
        self.showImagePre.setStyleSheet("border: 0px")

        self.softExit = QPushButton("退出")
        self.softExit.setFixedSize(45, 28)
        self.softExit.setStyleSheet("border: 0px")

        self.upImage = QPushButton("+")
        self.upImage.setFixedSize(23, 23)
        self.upImage.setStyleSheet(''' 
                     QPushButton
                     {
                     border-color: gray;
                     border-width: 1px;
                     border-radius: 11px;
                     border-style: outset;
                     }
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 11px;
                     border-style: outset;
                     }
                     ''')

        self.downImage = QPushButton("-")
        self.downImage.setFixedSize(23, 23)
        self.downImage.setStyleSheet(''' 
                     QPushButton
                     {
                     border-color: gray;
                     border-width: 1px;
                     border-radius: 11px;
                     border-style: outset;
                     }
                     QPushButton:pressed
                     {text-align : center;
                     background-color : light gray;
                     border-color: gray;
                     border-width: 2px;
                     border-radius: 11px;
                     border-style: outset;
                     }
                     ''')

        self.showImgSpe = QLabel(self)
        self.showImgSpe.setMaximumSize(self.screenWidth-30, self.screenHeight-150)
        self.showImgSpe.setStyleSheet("border: 1px solid gray")
        #self.showImgSpe.setFixedSize(500,500)

        self.blurredImageOpen.clicked.connect(lambda: self.open_file())
        self.saveImage.clicked.connect(self.save_image)
        self.deblurImage.clicked.connect(self.img_deblur)
        self.softExit.clicked.connect(self.close)
        self.upImage.clicked.connect(self.up_scale)
        self.downImage.clicked.connect(self.down_scale)
        self.showImagePre.pressed.connect(self.show_blur_image)
        self.showImagePre.released.connect(self.show_sharp_image)

        HLayout1 = QHBoxLayout()
        HLayout2 = QHBoxLayout()

        HLayout1.addWidget(self.blurredImageOpen)
        HLayout1.addWidget(self.saveImage)
        HLayout1.addWidget(self.deblurImage)
        HLayout1.addWidget(self.showImagePre)
        HLayout1.addWidget(self.softExit, 0, Qt.AlignLeft)
        HLayout1.addWidget(self.upImage, 0, Qt.AlignRight)
       # HLayout1.addWidget(self.showScale)
        HLayout1.addWidget(self.downImage)
        #HLayout1.setSpacing(0)

        HLayout2.addWidget(self.showImgSpe)

        mainLayout = QVBoxLayout()
        mainLayout.addLayout(HLayout1)
        mainLayout.addLayout(HLayout2)

        self.setLayout(mainLayout)
        self.setWindowTitle("模糊图像复原")
        self.resize(self.winSize[0], self.winSize[1])
        self.center()
        #self.setMaximumSize(1024,720)
        self.show()

    def center(self):
        #screen = QDesktopWidget().screenGeometry()
        size = self.geometry()
        self.move((self.screenWidth - size.width()) / 2, 0)
        # print(size.width(), size.height() )

    def open_file(self):
        try:
            path = QFileDialog.getOpenFileName(self, "Open Image", "./img", "PNG files(*.png);;JPG files(*.jpg)")
            self.blurredImagePath = str(path[0])
            img = np.array(Image.open(self.blurredImagePath)).astype(np.uint8)
            #print(img.shape)
            imgSize = img.shape
            blurImg = QPixmap(str(path[0]))
            # print(type(blurImg))
            if imgSize[0] > self.winSize[1] or imgSize[1] > self.winSize[0]:
                self.showImgSpe.setPixmap(blurImg.scaled(self.winSize[0]-30, self.winSize[1]-70, Qt.KeepAspectRatio))
                self.openSize = [self.winSize[1]-70, self.winSize[0]-30]
            else:
                self.showImgSpe.setPixmap(blurImg)
                self.openSize =  imgSize
            self.showImgSpe.setAlignment(Qt.AlignCenter)
            self.showImage = img
            self.blurredImage = img
            self.sharpImage = np.array(())
            self.showSize = self.openSize
            self.scale = 1
        except:
            pass

    def save_image(self):
        try:
            path = QFileDialog.getSaveFileName(self, "Open Image", "./img", "PNG files(*.png);;JPG files(*.jpg)")
            if self.showImage.shape[2] == 1:
                image_numpy = np.reshape(self.showImage, (self.showImage.shape[0], self.showImage.shape[1]))
                image = Image.fromarray(image_numpy, 'L')
            else:
                image = Image.fromarray(self.showImage)
            image.save(str(path[0]+'.png'))
        except:
            pass

    def up_scale(self):
        try:
            if self.scale < 3:
                self.scale = round(self.scale+0.1, 1)
                #print(self.scale)
            self.showSize = [self.openSize[0] * self.scale, self.openSize[1] * self.scale]
            self._show_image(self.showImage, self.showSize)
        except:
            pass

    def down_scale(self):
        try:
            if self.scale > 0.1:
                self.scale = round(self.scale-0.1, 1)
                #print((self.scale))
            self.showSize = [self.openSize[0]*self.scale, self.openSize[1]*self.scale]
            self._show_image(self.showImage, self.showSize)
        except:
            pass

    def img_deblur(self):
        try:
            deblur_net = deblur.DeblurNet()
            deblur_net.load_state_dict(torch.load('./model/deblur_shuffle_se_resizeconv.pth'))

            blur_img = Image.open(self.blurredImagePath)
            blur_img = blur_img.resize((528,(528*blur_img.size[1])//blur_img.size[0]), Image.BICUBIC)
            blur_img = deblur.image_transform(blur_img)
            blur_img = Variable(blur_img.unsqueeze(0))

            sharp_img = deblur_net(blur_img)
            sharp_img = deblur.image_recovery(sharp_img)

            self._show_image(sharp_img, self.showSize)
            self.sharpImage = sharp_img
            self.showImage = sharp_img
            #print("success")
        except:
            pass
            #print('False')

    def show_blur_image(self):
        try:
            self._show_image(self.blurredImage, self.showSize)
        except:
            pass

    def show_sharp_image(self):
        try:
            self._show_image(self.sharpImage, self.showSize)
        except:
            pass

    def _show_image(self, img, scale_size=[]):
        show_img = np.array(img).copy()
        show_img = QImage(show_img.data, show_img.shape[1], show_img.shape[0], QImage.Format_RGB888)
        if not scale_size:
            self.showImgSpe.setPixmap(QPixmap.fromImage(show_img))
        else:
            self.showImgSpe.setPixmap(QPixmap.fromImage(show_img).scaled(scale_size[1], scale_size[0], Qt.KeepAspectRatio))
        self.showImgSpe.setAlignment(Qt.AlignCenter)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    deblur_UI = Deblur_UI()
    sys.exit(app.exec())

