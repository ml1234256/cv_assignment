import torch
import torch.nn as nn
from torch.autograd import Variable
from torchvision import transforms as tfs
import numpy as np
from PIL import Image
import torch.nn.functional as F

def deblur(img_path):
    deblur_net = DeblurNet()
    deblur_net.load_state_dict(torch.load('deblur_net.pth'))

    blur_img = Image.open(img_path)


    blur_img = image_transform(blur_img)
    print(blur_img.shape)
    blur_img = Variable(blur_img.unsqueeze(0))

    sharp_img = deblur_net(blur_img)
    sharp_img = image_recovery(sharp_img)

    return sharp_img

def image_transform(x):
    transform_list = []
    transform_list += [tfs.ToTensor(),
                       tfs.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    transform = tfs.Compose(transform_list)
    x = transform(x)
    return x

def image_recovery(image_tensor):
    image_numpy = image_tensor[0].cpu().float().detach().numpy()
    image_numpy = (np.transpose(image_numpy, (1, 2, 0)) + 1) / 2.0 * 255.0
    return image_numpy.astype(np.uint8)


class DeblurNet(nn.Module):
    def __init__(self):
        super(DeblurNet, self).__init__()

        deblur_model = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, 64, 7, padding=0),
            nn.InstanceNorm2d(64, track_running_stats=True),
            nn.ReLU(True)
        ]

        deblur_model += [
            nn.Conv2d(64, 128, 3, 2, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=True),
            nn.ReLU(True),
            nn.Conv2d(128, 256, 3, 2, padding=1),
            nn.InstanceNorm2d(256, track_running_stats=True),
            nn.ReLU(True)
        ]

        for i in range(9):

            deblur_model += [
                ShuffleNetV1Unit1(256, 256)
            ]

        deblur_model += [
            #nn.ConvTranspose2d(256, 128, 3, 2, padding=1, output_padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(256, 128, 3, padding=1),
            nn.InstanceNorm2d(128, track_running_stats=True),
            nn.ReLU(True),

            #nn.ConvTranspose2d(128, 64, 3, 2, padding=1, output_padding=1),
            nn.UpsamplingBilinear2d(scale_factor=2),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.InstanceNorm2d(64, track_running_stats=True),
            nn.ReLU(True),
        ]

        deblur_model += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(64, 3, 7, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*deblur_model)

    def forward(self, x):
        res = x
        out = self.model(x)

        return torch.clamp(out + res, min=-1, max=1)


########################### ShuffleNet-V1 ##############################
class Hsigmoid(nn.Module):
   def __init__(self, inplace=True):
      super(Hsigmoid, self).__init__()
      self.inplace = inplace

   def forward(self, x):
      out = F.relu6(x + 3, inplace=self.inplace) / 6
      return out

class SeLayer(nn.Module):
   def __init__(self, dim, reduction=4):
      super(SeLayer, self).__init__()
      self.se = nn.Sequential(
         nn.AdaptiveAvgPool2d(1),
          nn.Conv2d(dim, dim//reduction, kernel_size=1, stride=1, padding=0, bias=False),
          nn.ReLU(inplace=True),
          nn.Conv2d(dim//reduction, dim, kernel_size=1, stride=1, padding=0, bias=False),
          nn.Sigmoid()
      )
   def forward(self, x):
      return x * self.se(x)


class Channel_Shuffle(nn.Module):
    def __init__(self, groups):
        super(Channel_Shuffle, self).__init__()
        self.groups = groups

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        assert channels % self.groups == 0
        channels_per_group = channels // self.groups
        # split into groups
        x = x.view(batch_size, self.groups, channels_per_group, height, width)
        # transpose 1, 2 axis
        x = x.transpose(1, 2).contiguous()
        # reshape into original
        x = x.view(batch_size, channels, height, width)
        return x

class ShuffleNetV1Unit1(nn.Module):
    """ShuffleNet unit for stride=1"""
    def __init__(self, in_channels, out_channels, groups=2):
        super(ShuffleNetV1Unit1, self).__init__()
        assert in_channels == out_channels
        assert out_channels % 4 ==0
        bottleneck_channels = out_channels // 4

        self.shufflenet_block = nn.Sequential(
            # 1x1 GConv
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0,groups=groups),
            nn.InstanceNorm2d(bottleneck_channels),
            nn.ReLU(True),
            # channel shuffle
            Channel_Shuffle(groups=groups),
            # 3x3 DWConv
            nn.ReflectionPad2d(1),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=0, stride=1,
                      groups=bottleneck_channels),
            nn.InstanceNorm2d(bottleneck_channels),
           # nn.ReLU(True),
            # 1x1 GConv
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups),
            nn.InstanceNorm2d(out_channels),

            # 1x1 GConv
            nn.Conv2d(in_channels, bottleneck_channels, kernel_size=1, stride=1, padding=0, groups=groups),
            nn.InstanceNorm2d(bottleneck_channels),
            nn.ReLU(True),
            # channel shuffle
            Channel_Shuffle(groups=groups),
            # 3x3 DWConv
            nn.ReflectionPad2d(1),
            nn.Conv2d(bottleneck_channels, bottleneck_channels, kernel_size=3, padding=0, stride=1,
                      groups=bottleneck_channels),
            nn.InstanceNorm2d(bottleneck_channels),
            #nn.ReLU(True),
            # 1x1 GConv
            nn.Conv2d(bottleneck_channels, out_channels, kernel_size=1, stride=1, padding=0, groups=groups),
            nn.InstanceNorm2d(out_channels)
        )

       # self.shufflenet_block.apply(gaussian_weights_init)

        self.relu = nn.ReLU(True)
        self.se = SeLayer(out_channels)

    def forward(self, x):
        out = self.shufflenet_block(x)
        out = x + self.se(out)
       # out = x + out
        return out

if __name__ == '__main__':
    deblur_net = DeblurNet()
    print(deblur_net)
    deblur_net.load_state_dict(torch.load('./model/deblur_shuffle_se_resizeconv.pth'))

    # import matplotlib.pyplot as plt
    # img = deblur('./img/test1.png')
    # plt.figure()
    # plt.imshow(img)
    # plt.pause(0)

