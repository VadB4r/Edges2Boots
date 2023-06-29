import torch
import torch.nn as nn

class Generator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1), #256->256
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), #256->256
            nn.BatchNorm2d(64),
            nn.PReLU(64)
        )
        self.pool0 =  nn.MaxPool2d(2) #256->128
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), #128->128
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1), #128->128
            nn.BatchNorm2d(128),
            nn.PReLU(128)
        )
        self.pool1 =  nn.MaxPool2d(2) #128->64
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), #64->64
            nn.BatchNorm2d(256),
            nn.PReLU(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1), #64->64
            nn.BatchNorm2d(256),
            nn.PReLU(256)
        )
        self.pool2 =  nn.MaxPool2d(2) #64->32
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #32->32
            nn.BatchNorm2d(512),
            nn.PReLU(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), #32->32
            nn.BatchNorm2d(512),
            nn.PReLU(512)
        )
        self.pool3 =  nn.MaxPool2d(2) #32->16

        # bottleneck
        self.bottleneck_conv = nn.Sequential(
            nn.Conv2d(512, 1024, kernel_size=3, padding=1), #16->16
            nn.BatchNorm2d(1024),
            nn.PReLU(1024),
            nn.Conv2d(1024, 1024, kernel_size=3, padding=1), #16->16
            nn.BatchNorm2d(1024),
            nn.PReLU(1024)
        )

        # decoder (upsampling)
        self.upsample0 = nn.Upsample(scale_factor=2, mode='bilinear') #16->32
        self.cut0 = nn.Conv2d(1024, 512, kernel_size=1)
        self.dec_conv0 = nn.Sequential(
            nn.Conv2d(1024, 512, kernel_size=3, padding=1), #32->32
            nn.BatchNorm2d(512),
            nn.PReLU(512),
            nn.Conv2d(512, 512, kernel_size=3, padding=1), #32->32
            nn.BatchNorm2d(512),
            nn.PReLU(512)
        )
        self.upsample1 = nn.Upsample(scale_factor=2, mode='bilinear') #32->64
        self.cut1 = nn.Conv2d(512, 256, kernel_size=1)
        self.dec_conv1 = nn.Sequential(
            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(256),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.PReLU(256)
        )
        self.upsample2 = nn.Upsample(scale_factor=2, mode='bilinear') #64->128
        self.cut2 = nn.Conv2d(256, 128, kernel_size=1)
        self.dec_conv2 = nn.Sequential(
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(128),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.PReLU(128)
        )
        self.upsample3 = nn.Upsample(scale_factor=2, mode='bilinear') #128->256
        self.cut3 = nn.Conv2d(128, 64, kernel_size=1)
        self.dec_conv3 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1), #256->256
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Conv2d(64, 64, kernel_size=3, padding=1), #256->256
            nn.BatchNorm2d(64),
            nn.PReLU(64),
            nn.Conv2d(64, 3, kernel_size=1)
        )

    def forward(self, x):
        # encoder
        e0 = self.enc_conv0(x)
        p0 = self.pool0(e0)
        e1 = self.enc_conv1(p0)
        p1 = self.pool1(e1)
        e2 = self.enc_conv2(p1)
        p2 = self.pool2(e2)
        e3 = self.enc_conv3(p2)
        p3 = self.pool3(e3)

        # bottleneck
        b = self.bottleneck_conv(p3)
        
        # decoder
        u0 = self.upsample0(self.cut0(b))
        d0 = self.dec_conv0(torch.cat((e3, u0), 1))
        u1 = self.upsample1(self.cut1(d0))
        d1 = self.dec_conv1(torch.cat((e2, u1), 1))
        u2 = self.upsample2(self.cut2(d1))
        d2 = self.dec_conv2(torch.cat((e1, u2), 1))
        u3 = self.upsample3(self.cut3(d2))
        d3 = self.dec_conv3(torch.cat((e0, u3), 1))
        return d3.tanh()