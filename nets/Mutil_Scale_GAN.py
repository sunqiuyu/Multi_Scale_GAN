import torch
import torch.nn as nn
from torchvision import models

class DecoderBlock(nn.Module):
    def __init__(self, in_channels, n_filters):
        super(DecoderBlock,self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels // 4, 1)
        self.norm1 = nn.BatchNorm2d(in_channels // 4)
        self.drop = nn.Dropout2d(0.3)
        self.relu1 = nn.ReLU(inplace=True)

        self.deconv2 = nn.ConvTranspose2d(in_channels // 4, in_channels // 4, 3, stride=2, padding=1, output_padding=1)
        self.norm2 = nn.BatchNorm2d(in_channels // 4)
        self.relu2 = nn.ReLU(inplace=True)

        self.conv3 = nn.Conv2d(in_channels // 4, n_filters, 1)
        self.norm3 = nn.BatchNorm2d(n_filters)
        self.relu3 = nn.ReLU(inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.drop(x)
        x = self.relu1(x)
        x = self.deconv2(x)
        x = self.norm2(x)
        x = self.drop(x)
        x = self.relu2(x)
        x = self.conv3(x)
        x = self.norm3(x)
        x = self.drop(x)
        x = self.relu3(x)
        return x

#解码层
class Decoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        #反卷积，使之可以和编码层拼起来
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        # residual function
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut 捷径，跳跃连接的意思
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x, encoder_x):
        x = self.transconv(x)

        #cat会使两个张量的通道数相加
        x = torch.cat([x, encoder_x], dim=1)
        return nn.ReLU(inplace=True)(self.residual_function(x) + self.shortcut(x))

class BasicBlock(nn.Module):
    """Basic Block for resnet 18 and resnet 34
    """

    # BasicBlock and BottleNeck block
    # have different output size
    # we use class attribute expansion
    # to distinct
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1,  reduction=16):
        super().__init__()

        # residual function
        #这层卷积块后期可以替换为resnet有三个卷积的残差结构
        self.residual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut 捷径，跳跃连接的意思
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

        #SE layers 通道注意力机制  1*1*（c/r）r=16

        # Squeeze 对任意尺寸图片可以指定输出大小，不会改变通道数， 1*1*C
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        #Excitation
        self.Excitation = nn.Sequential(
            nn.Linear(out_channels * BasicBlock.expansion, out_channels * BasicBlock.expansion//reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(out_channels * BasicBlock.expansion//reduction, out_channels * BasicBlock.expansion, bias=False),
            nn.Sigmoid()
        )


    def forward(self, x):
        x_r = self.residual_function(x)
        b, c, _, _ = x_r.size()
        w = self.avg_pool(x_r).view(b, c)
        w = self.Excitation(w).view(b, c, 1, 1)
        #print("x_r=",x_r.shape)
        #print("w=", w.shape)
        x_w = x_r * w
        #print("x_w=", x_w.shape)
        return nn.ReLU(inplace=True)(x_w + self.shortcut(x))

#中间层——扩张残差模块
class MidUnet(nn.Module):
    def __init__(self,in_channels, out_channels, stride=1):
        super().__init__()
        # residual function
        self.DilatedResidual_function = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=2, dilation=2, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels * BasicBlock.expansion, kernel_size=3, padding=3, dilation=3,  bias=False),
            nn.BatchNorm2d(out_channels * BasicBlock.expansion)
        )

        # shortcut 捷径，跳跃连接的意思
        self.shortcut = nn.Sequential()

        # the shortcut output dimension is not the same with residual function
        # use 1*1 convolution to match the dimension
        if stride != 1 or in_channels != BasicBlock.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels * BasicBlock.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels * BasicBlock.expansion)
            )

    def forward(self, x):
        return nn.ReLU(inplace=True)(self.DilatedResidual_function(x) + self.shortcut(x))


class GeneratorUNet_4scale(nn.Module):
    def __init__(self, num_classes = 1):
        super(GeneratorUNet_4scale, self).__init__()
        #参数都是通道数
        self.encoder1 = BasicBlock(3, 64)
        self.encoder2 = BasicBlock(64, 128)
        self.encoder3 = BasicBlock(128, 256)
        self.encoder4 = BasicBlock(256, 512)

        #最底层使用扩张残差块
        self.encoder5 = MidUnet(512, 1024)

        self.down_sample = nn.MaxPool2d(2)

        self.deconder1 = Decoder(1024, 512)
        self.deconder2 = Decoder(512, 256)
        self.deconder3 = Decoder(256, 128)
        self.deconder4 = Decoder(128, 64)

        self.transconv1 = nn.ConvTranspose2d(512, 256, 4, 2, 1)
        self.transconv2 = nn.ConvTranspose2d(512, 128, 4, 2, 1)
        self.transconv3 = nn.ConvTranspose2d(256, 64, 4, 2, 1)

        self.layer1 = nn.Conv2d(512,num_classes,1)
        self.layer2 = nn.Conv2d(256, num_classes, 1)
        self.layer3 = nn.Conv2d(128, num_classes, 1)


        #输出图片
        self.lastconv = nn.Sequential(
            nn.Conv2d(64,32,kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32,num_classes,1)
        )

    def forward(self, x):
        x1 = self.encoder1(x)
        x = self.down_sample(x1)

        x2 = self.encoder2(x)
        x = self.down_sample(x2)

        x3 = self.encoder3(x)
        x = self.down_sample(x3)

        x4 = self.encoder4(x)
        x = self.down_sample(x4)

        x5 = self.encoder5(x)

        #三个尺度的深度监督, _ds是要用于深度监督的
        x6 = self.deconder1(x5, x4)

        x7 = self.deconder2(x6, x3)

        x8 = self.deconder3(x7, x2)


        x9 = self.deconder4(x8, x1)

        output = self.lastconv(x9)
        output1 = self.layer1(x6)
        output2 = self.layer2(x7)
        output3 = self.layer3(x8)

        return output,output3,output2,output1

#*************************************************#
#                判别网络
#*************************************************#
class D_Encoder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        self.Layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.Dropout2d(0.3),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.Layers(x)

class D_Deconder(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # 反卷积，使之可以和编码层拼起来
        self.transconv = nn.ConvTranspose2d(in_channels, out_channels, 4, 2, 1)
        # residual function
        self.Layers = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x, encoder_x):
        x = self.transconv(x)
        # cat会使两个张量的通道数相加
        x = torch.cat([x, encoder_x], dim=1)
        return nn.ReLU(inplace=True)(self.Layers(x))

class DiscriminatorUNet(nn.Module):
    def __init__(self, num_classes = 1):
        super(DiscriminatorUNet, self).__init__()

        #参数都是通道数
        self.encoder1 = D_Encoder(3 + num_classes, 32)
        self.encoder2 = D_Encoder(32, 64)
        self.encoder3 = D_Encoder(64, 128)
        self.encoder4 = D_Encoder(128, 256)
        self.encoder5 = D_Encoder(256, 512)

        self.down_sample = nn.MaxPool2d(2)

        self.deconder1 = D_Deconder(512, 256)
        self.deconder2 = D_Deconder(256, 128)
        self.deconder3 = D_Deconder(128, 64)
        self.deconder4 = D_Deconder(64, 32)

        #输出图片
        self.lastconv = nn.Conv2d(32, 1, 1)

    def forward(self, x,label):
        x = torch.cat([x,label],1)
        x1 = self.encoder1(x)
        x = self.down_sample(x1)

        x2 = self.encoder2(x)
        x = self.down_sample(x2)

        x3 = self.encoder3(x)
        x = self.down_sample(x3)

        x4 = self.encoder4(x)
        x = self.down_sample(x4)

        x5 = self.encoder5(x)

        x = self.deconder1(x5, x4)
        x = self.deconder2(x, x3)
        x = self.deconder3(x, x2)
        x = self.deconder4(x, x1)

        #输出的卷积
        out = self.lastconv(x)

        return torch.sigmoid(out)

if __name__ == '__main__':
    device = torch.device('cpu')
    x = torch.randn(2, 3, 256, 256).to(device)
    x1 = torch.ones(2, 2, 256, 256).to(device)
    net = GeneratorUNet_4scale(num_classes=2).to(device)
    x2 = net(x)
    print(x2[0].shape)

