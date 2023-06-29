import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc_conv0 = nn.Sequential(
            nn.Conv2d(6, 64, kernel_size=3, padding=1), #256->256
            nn.BatchNorm2d(64),
            nn.PReLU(64),
        )
        self.pool0 =  nn.MaxPool2d(2) #256->128
        self.enc_conv1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1), #128->128
            nn.BatchNorm2d(128),
            nn.PReLU(128),
        )
        self.pool1 =  nn.MaxPool2d(2) #128->64
        self.enc_conv2 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, padding=1), #64->64
            nn.BatchNorm2d(256),
            nn.PReLU(256),
        )
        self.pool2 =  nn.MaxPool2d(2) #64->32
        self.enc_conv3 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, padding=1), #32->32
            nn.BatchNorm2d(512),
            nn.PReLU(512),
        )
        self.pool3 =  nn.MaxPool2d(2) #32->16
        self.enc_conv4 = nn.Sequential(
            nn.Conv2d(512, 1, kernel_size=1), #32->32
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
        e4 = self.enc_conv4(p3)
        
        return e4.sigmoid()